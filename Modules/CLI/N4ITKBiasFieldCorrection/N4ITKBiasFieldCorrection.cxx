#include "itkBinaryBallStructuringElement.h"
#include "itkFlatStructuringElement.h"

#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryFillholeImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkFlipImageFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageFileWriter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkOtsuThresholdImageCalculator.h"
#include "itkReconstructionByDilationImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkStatisticsImageFilter.h"

#include "vtkMath.h"

#if ITK_VERSION_MAJOR >= 4
// This is  now officially part of ITKv4
#include "itkN4BiasFieldCorrectionImageFilter.h"
#else
// Need private version for ITKv3 that does not conflict with ITKv4 fixes
#include "SlicerITKv3N4MRIBiasFieldCorrectionImageFilter.h"
#endif

#include "N4ITKBiasFieldCorrectionCLP.h"
#include "itkPluginUtilities.h"

#define MAX(x,y) (((x) > (y)) ? x : y)

namespace
{

/* allocate memory for an nrow x ncol matrix */
template<class TReal>
TReal **create_matrix ( long nrow, long ncol )
{
  typedef TReal* TRealPointer;
  TReal **m = new TRealPointer[nrow];

  TReal* block = ( TReal* ) calloc ( nrow*ncol, sizeof ( TReal ) );
  m[0] = block;
  for ( int row = 1; row < nrow; ++row )
    {
	m[ row ] = &block[ row * ncol ];
    }
  return m;
}

/* free a TReal matrix allocated with matrix() */
template<class TReal>
void free_matrix ( TReal **m )
{
  free ( m[0] );
  delete[] m;
}

typedef std::vector<double> doublevec;
typedef std::pair<doublevec *, doublevec *> histogrampair;

template<class TImage>
histogrampair * create_histogram_aux( const TImage * inputImage, int numbins, bool uselowerthreshold, typename TImage::PixelType lowerthreshold, bool useupperthreshold, typename TImage::PixelType upperthreshold)
{
  // histogram code (from OtsuImageThreshold)
  double totalPixels = (double) inputImage->GetBufferedRegion().GetNumberOfPixels();
  if ( totalPixels == 0 ) {
    std::cout << "ERROR: empty image!" << std::endl;
    return NULL;
  }
  // compute image max and min
  typedef itk::MinimumMaximumImageCalculator<TImage> RangeCalculator;
  typename RangeCalculator::Pointer rangeCalculator = RangeCalculator::New();
  rangeCalculator->SetImage( inputImage );
  rangeCalculator->Compute();

  typename TImage::PixelType imageMin = rangeCalculator->GetMinimum();
  typename TImage::PixelType imageMax = rangeCalculator->GetMaximum();

  if ( imageMin >= imageMax ) {
    return NULL;
  }

  if (uselowerthreshold) {
    if ( imageMin < lowerthreshold ) {
      imageMin = lowerthreshold;
    }
  } else {
    lowerthreshold = imageMin;
  }

  if (useupperthreshold) {
    if ( imageMax > upperthreshold ) {
      imageMax = upperthreshold;
    }
  } else {
    upperthreshold = imageMax;
  }

  double binMultiplier = (double) numbins /
    (double) ( imageMax - imageMin );
  double binWidth = (double) ( imageMax - imageMin ) /
    (double) numbins;

  doublevec * relativeFrequencyPtr = new doublevec();
  doublevec & relativeFrequency = *relativeFrequencyPtr;
  relativeFrequency.resize( numbins );
  doublevec * binBoundariesPtr = new doublevec();
  doublevec & binBoundaries = *binBoundariesPtr;
  binBoundaries.resize( numbins + 1);
  for ( int j = 0; j < numbins; j++ )
    {
      relativeFrequency[j] = 0.0;
      binBoundaries[j] = imageMin + (binWidth * j);
    }
  binBoundaries[numbins] = imageMin + (binWidth * numbins);

  typedef itk::ImageRegionConstIteratorWithIndex<TImage> Iterator;
  Iterator iter( inputImage, inputImage->GetBufferedRegion() );

  totalPixels = 0;
  while ( !iter.IsAtEnd() )
    {
      int binNumber;
      typename TImage::PixelType value = iter.Get();
      ++iter;

      if ( value < lowerthreshold )
	{
	  continue;
	}
      else if ( value > upperthreshold )
	{
	  continue;
	}
      else if ( value == imageMin ) 
	{
	  binNumber = 0;
	}
      else
	{
	  binNumber = (int) ceil( (value - imageMin) * binMultiplier ) - 1;
	  if ( binNumber == numbins ) // in case of rounding errors
	    {
	      binNumber -= 1;
	    }
	}

      totalPixels++;
      relativeFrequency[binNumber] += 1.0;

    }
  // normalize the frequencies
  for ( int j = 0; j < numbins; j++ )
    {
      relativeFrequency[j] /= totalPixels;
    }

  return new histogrampair(relativeFrequencyPtr, binBoundariesPtr);
}

template<class TImage>
histogrampair * create_histogram( const TImage * inputImage, int numbins, typename TImage::PixelType upperthreshold, typename TImage::PixelType lowerthreshold)
{
  return create_histogram_aux(inputImage, numbins, true, upperthreshold, true, lowerthreshold);
}

template<class TImage>
histogrampair * create_histogram( const TImage * inputImage, int numbins)
{
  return create_histogram_aux(inputImage, numbins, false, 0, false, 0);
}

void
print_histogram(double * h, int numbins, double * y = NULL) {
  // find last "significant" bin (just for display)
  int lastsigbin;
  for ( lastsigbin = numbins - 1; lastsigbin >= 0; lastsigbin-- ) {
    if (h[lastsigbin] > 0.0001) {
      break;
    }
  }
  double maxbinheight = h[0];
  for (int j = 1; j < lastsigbin; j++) {
    if (h[j] > maxbinheight) {
      maxbinheight = h[j];
    }
  }
  // last "significant" bin changes now that we know what scale at which we'll print the histogram
  for ( /* null */ ; lastsigbin >= 0; lastsigbin-- ) {
    if (50 * (h[lastsigbin] / maxbinheight) >= 1) {
      break;
    }
  }
  std::string spacestr = std::string(50, '-');
  for (int j = 0; j < lastsigbin; j++) {
    int height = 50 * (h[j] / maxbinheight);
    if (y == NULL) {
      std::cout << "    " << setw(5) << j << &spacestr.c_str()[50 - height] << '*' << std::endl;
      continue;
    }
    int fitheight = 50 * (y[j] / maxbinheight);
    if (fitheight < 0) {
      fitheight = 0;
    }
    if (height == fitheight) {
      std::cout << "    " << setw(5) << j << &spacestr.c_str()[50 - height] << '*' << std::endl;
    } else if (height < fitheight) {
      std::cout << "    " << setw(5) << j << &spacestr.c_str()[50 - height] << '#' << &spacestr.c_str()[50 - (fitheight - height - 1)] << '@' << std::endl;
    } else {
      std::cout << "    " << setw(5) << j << &spacestr.c_str()[50 - fitheight] << '@' << &spacestr.c_str()[50 - (height - fitheight - 1)] << '#' << std::endl;
    }
  }
}

void
print_histogram(const std::vector<double> & h, int numbins, double * y = NULL) {
  double * hptr = (double *)malloc(sizeof(double) * numbins);
  for (int i = 0; i < numbins; i++) {
    hptr[i] = h[i];
  }
  print_histogram(hptr, numbins, y);
  free(hptr);
}

typedef float RealType;
const int ImageDimension = 3;
typedef itk::Image<RealType, ImageDimension> ImageType;

template <class TFilter>
class CommandIterationUpdate : public itk::Command
{
public:
  typedef CommandIterationUpdate  Self;
  typedef itk::Command            Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  itkNewMacro( Self );
protected:
  CommandIterationUpdate()
  {
  };
public:

  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *) caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    const TFilter * filter =
      dynamic_cast<const TFilter *>( object );

    if( typeid( event ) != typeid( itk::IterationEvent ) )
      {
      return;
      }
    std::cout << "Progress: " << filter->GetProgress() << std::endl;
  }

};

template <class T>
int SaveIt(ImageType::Pointer img, const char* fname, T)
{
  typedef itk::Image<T, 3>                                 OutputImageType;
  typedef itk::CastImageFilter<ImageType, OutputImageType> CastType;

  typename CastType::Pointer caster = CastType::New();
  caster->SetInput(img);

  typedef  itk::ImageFileWriter<OutputImageType> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput( caster->GetOutput() );
  writer->SetFileName( fname );
  writer->SetUseCompression(1);
  writer->Update();

  return EXIT_SUCCESS;
}

};

int main(int argc, char* * argv)
{

  PARSE_ARGS;

  ImageType::Pointer inputImage = NULL;

  typedef itk::Image<unsigned char, ImageDimension> MaskImageType;
  MaskImageType::Pointer maskImage = NULL;

#if ITK_VERSION_MAJOR >= 4
  typedef    itk::N4BiasFieldCorrectionImageFilter<ImageType, MaskImageType, ImageType> CorrecterType;
#else
  typedef itk::N4MRIBiasFieldCorrectionImageFilter<ImageType, MaskImageType, ImageType> CorrecterType;
#endif
  CorrecterType::Pointer correcter = CorrecterType::New();

  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName( inputImageName.c_str() );
  reader->Update();
  inputImage = reader->GetOutput();

  /**
   * handle he mask image
   */

  if( maskImageName != "" )
    {
    typedef itk::ImageFileReader<MaskImageType> ReaderType;
    ReaderType::Pointer maskreader = ReaderType::New();
    maskreader->SetFileName( maskImageName.c_str() );
    maskreader->Update();
    maskImage = maskreader->GetOutput();
    itk::ImageRegionConstIterator<MaskImageType> IM(
      maskImage, maskImage->GetBufferedRegion() );
    MaskImageType::PixelType maskLabel = 0;
    for( IM.GoToBegin(); !IM.IsAtEnd(); ++IM )
      {
      if( IM.Get() )
        {
        maskLabel = IM.Get();
        break;
        }
      }
    if( !maskLabel )
      {
      return EXIT_FAILURE;
      }
    correcter->SetMaskLabel(maskLabel);
    }

  if ( newMask )
    {
      std::cout << "Creating new mask." << std::endl;

      std::cout << "  Finding proper threshold..." << std::endl;

      // compute image max and min
      typedef itk::MinimumMaximumImageCalculator<ImageType> RangeCalculator;
      RangeCalculator::Pointer rangeCalculator = RangeCalculator::New();
      rangeCalculator->SetImage( inputImage );
      rangeCalculator->Compute();

      ImageType::PixelType imageMin = rangeCalculator->GetMinimum();
      ImageType::PixelType imageMax = rangeCalculator->GetMaximum();

      std::cout << "    Image min, max: (" << imageMin << ", " << imageMax << ")" << std::endl;

      int numberOfHistogramBins = 100;
      int polyorder = 10;

      // get initial histogram
      histogrampair * histogramPtr1 = create_histogram<ImageType>(inputImage, numberOfHistogramBins, imageMin, imageMax);
      doublevec & relativeFrequency1 = *(histogramPtr1->first);
      doublevec & binBoundaries1 = *(histogramPtr1->second);
      ImageType::PixelType binWidth1 = binBoundaries1[1] - binBoundaries1[0];
      std::cout << "Raw histogram (bin width " << binWidth1 << "):" << std::endl;
      print_histogram(relativeFrequency1, numberOfHistogramBins);
      // find last "significant" bin
      int lastsigbin;
      for ( lastsigbin = numberOfHistogramBins - 1; lastsigbin >= 0; lastsigbin-- ) {
	if (relativeFrequency1[lastsigbin] > 0.001) {
	  break;
	}
      }
      // get new histogram excluding the insignificant tail
      histogrampair * histogramPtr2 = create_histogram<ImageType>(inputImage, numberOfHistogramBins, imageMin, (*(histogramPtr1->second))[lastsigbin + 1]);
      doublevec & relativeFrequency2 = *(histogramPtr2->first);
      doublevec & binBoundaries2 = *(histogramPtr2->second);

      delete histogramPtr1->first;
      delete histogramPtr1->second;
      delete histogramPtr1;
    

      // fit a curve to the histogram
      int numberOfSamples = numberOfHistogramBins;
      int numberOfVariables = polyorder;
      double **x = create_matrix<double> (numberOfSamples, numberOfVariables);
      for (int sn = 0; sn < numberOfSamples; sn++) {
	for (int vn = 0; vn < numberOfVariables; vn++) {
	  x[sn][vn] = pow((double)sn, (double)vn);
	}
      }
      double **m = create_matrix<double> ( numberOfVariables, 1 );
      double **y = create_matrix<double> ( numberOfSamples, 1 );
      for (int sn = 0; sn < numberOfSamples; sn++) {
	y[sn][0] = relativeFrequency2[sn];
      }
      vtkMath::SolveLeastSquares(numberOfSamples, x, numberOfVariables, y, 1, m);
      vtkMath::MultiplyMatrix((const double **)x, (const double **)m,
			      numberOfSamples, numberOfVariables,
			      numberOfVariables, 1,
			      y);
      // y now contains the fitted curve.
      ImageType::PixelType binWidth2 = binBoundaries2[1] - binBoundaries2[0];
      std::cout << "Cut-off histogram with fit (bin width " << binWidth2 << "):" << std::endl;
      print_histogram(relativeFrequency2, numberOfHistogramBins, y[0]);

      // double initthresh = imageMin;
      // for (int sn = 0; sn < numberOfSamples - 1; sn++) {
      // 	if (y[sn][0] <= y[sn+1][0]) {
      // 	  // found first upward slope -- choose boundary between bins as threshold
      // 	  std::cout << "    Found local minimum at index " << sn << " (values in surrounding range are [" << y[sn][0] << ", " << y[sn+1][0] << "])" << std::endl;
      // 	  initthresh = imageMin + ((sn + 1) * binWidth2);
      // 	  break;
      // 	}
      // }

      // start with Otsu threshold, but move it to a local minimum
      typedef itk::OtsuThresholdImageCalculator<ImageType> ThresholdCalculatorType;
      ThresholdCalculatorType::Pointer threshcalc = ThresholdCalculatorType::New();
      threshcalc->SetImage(inputImage);
      threshcalc->Compute();
      ImageType::PixelType initthresh = threshcalc->GetThreshold();
      int initbin = (int) ceil( (initthresh - imageMin) / binWidth2 ) - 1;
      std::cout << "    Starting with initial threshold " << initthresh << " (bin " << initbin << ")" << endl;
      // find the first downward contour of the fitted histogram backward
      // from the initial choice, then choose the threshold where the fit
      // goes back upwards.
      int changed = 0;
      while (initbin > 0 && y[initbin - 1][0] > y[initbin][0]) {
	initbin--;
      }
      std::cout << "    Moved backwards up to threshold bin " << initbin << std::endl;
      while (initbin > 0 && y[initbin - 1][0] <= y[initbin][0]) {
	initbin--;
	changed = 1;
      }
      std::cout << "    Moved backwards down to threshold bin " << initbin << std::endl;
      if (changed != 0) {
	initthresh = binBoundaries2[initbin];
      }

      delete histogramPtr2->first;
      delete histogramPtr2->second;
      delete histogramPtr2;

      free_matrix(x);
      free_matrix(m);
      free_matrix(y);

      std::cout << "    Initial threshold: " << initthresh << std::endl;

      typedef itk::BinaryThresholdImageFilter <ImageType, MaskImageType> ThresholdImageFilterType;
 
      ThresholdImageFilterType::Pointer thresholdFilter = ThresholdImageFilterType::New();
      thresholdFilter->SetInput(inputImage);
      thresholdFilter->SetLowerThreshold(initthresh);
      thresholdFilter->SetOutsideValue(0);
      thresholdFilter->SetInsideValue(1);
      thresholdFilter->Update();
      maskImage = thresholdFilter->GetOutput();


      typedef itk::BinaryBallStructuringElement<MaskImageType::PixelType, 3> BallType;

      MaskImageType::SpacingType maskspacing = maskImage->GetSpacing();
      std::cout << "  mask spacing: " << maskspacing << std::endl; \
      MaskImageType::SizeType size;
#define SETSIZE(radius) {						\
	size[0] = (maskspacing[0] > (radius)) ? 0: (int)((double)(radius) / maskspacing[0]); \
	size[1] = (maskspacing[1] > (radius)) ? 0: (int)((double)(radius) / maskspacing[1]); \
	size[2] = (maskspacing[2] > (radius)) ? 0: (int)((double)(radius) / maskspacing[2]); \
	std::cout << "  " << radius << "mm radius converted to size (in voxels): " << size << std::endl; \
      }

      BallType scrubfine;
      SETSIZE(2.0);
      scrubfine.SetRadius(size);
      scrubfine.CreateStructuringElement();
      BallType scrubcoarse;
      SETSIZE(8.0);
      scrubcoarse.SetRadius(size);
      scrubcoarse.CreateStructuringElement();
    
      typedef itk::BinaryErodeImageFilter <MaskImageType, MaskImageType, BallType> BinaryErodeImageFilterType;
      typedef itk::BinaryFillholeImageFilter <MaskImageType> BinaryFillholeImageFilterType;
      typedef itk::MaskImageFilter <MaskImageType, MaskImageType> MaskImageFilterType;

      if( outputMaskName != "" )
	{
	  typedef itk::ImageFileWriter<MaskImageType> WriterType;
	  WriterType::Pointer writer = WriterType::New();
	  writer->SetFileName( (std::string("origmask-") + outputMaskName).c_str() );
	  writer->SetInput( maskImage );
	  writer->SetUseCompression(1);
	  writer->Update();
	}

      std::cout << "  Filling holes in mask..." << std::endl;
      BinaryFillholeImageFilterType::Pointer fillFilter = BinaryFillholeImageFilterType::New();
      fillFilter->SetForegroundValue(1);
      fillFilter->SetInput(maskImage);
      fillFilter->Update();
      MaskImageType::Pointer filledmask = fillFilter->GetOutput();
      if( outputMaskName != "" )
	{
	  typedef itk::ImageFileWriter<MaskImageType> WriterType;
	  WriterType::Pointer writer = WriterType::New();
	  writer->SetFileName( (std::string("filledmask-") + outputMaskName).c_str() );
	  writer->SetInput( filledmask );
	  writer->SetUseCompression(1);
	  writer->Update();
	}

      std::cout << "  Eroding for mask..." << std::endl;
      BinaryErodeImageFilterType::Pointer erodeFilter1 = BinaryErodeImageFilterType::New();
      erodeFilter1->SetErodeValue(1);
      erodeFilter1->SetInput(filledmask);
      erodeFilter1->SetKernel(scrubfine);
      erodeFilter1->Update();
      MaskImageType::Pointer erodedmask = erodeFilter1->GetOutput();
      if( outputMaskName != "" )
	{
	  typedef itk::ImageFileWriter<MaskImageType> WriterType;
	  WriterType::Pointer writer = WriterType::New();
	  writer->SetFileName( (std::string("mask-") + outputMaskName).c_str() );
	  writer->SetInput( erodedmask );
	  writer->SetUseCompression(1);
	  writer->Update();
	}

      std::cout << "  Eroding for marker..." << std::endl;
      BinaryErodeImageFilterType::Pointer erodeFilter2 = BinaryErodeImageFilterType::New();
      erodeFilter2->SetErodeValue(1);
      erodeFilter2->SetInput(maskImage);
      erodeFilter2->SetKernel(scrubcoarse);
      erodeFilter2->Update();
      MaskImageType::Pointer markermask = erodeFilter2->GetOutput();
      if( outputMaskName != "" )
	{
	  typedef itk::ImageFileWriter<MaskImageType> WriterType;
	  WriterType::Pointer writer = WriterType::New();
	  writer->SetFileName( (std::string("marker-") + outputMaskName).c_str() );
	  writer->SetInput( markermask );
	  writer->SetUseCompression(1);
	  writer->Update();
	}

      std::cout << "  Reconstructing by dilation..." << std::endl;
      typedef itk::ReconstructionByDilationImageFilter<MaskImageType, MaskImageType> DilateFilterType;
      DilateFilterType::Pointer dilateFilter = DilateFilterType::New();
      dilateFilter->SetMarkerImage(markermask);
      dilateFilter->SetMaskImage(erodedmask);
      dilateFilter->Update();

      maskImage = dilateFilter->GetOutput();
    
    }

  if( !maskImage )
    {
    std::cout << "Mask no read.  Creaing Otsu mask." << std::endl;
    typedef itk::OtsuThresholdImageFilter<ImageType, MaskImageType>
	ThresholderType;
    ThresholderType::Pointer otsu = ThresholderType::New();
    otsu->SetInput( inputImage );
    otsu->SetNumberOfHistogramBins( 200 );
    otsu->SetInsideValue( 0 );
    otsu->SetOutsideValue( 1 );
    otsu->Update();

    maskImage = otsu->GetOutput();
    }

  if( outputMaskName != "" )
    {
    std::cout << "Writing output mask." << std::endl;
    typedef itk::ImageFileWriter<MaskImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( outputMaskName.c_str() );
    writer->SetInput( maskImage );
    writer->SetUseCompression(1);
    writer->Update();
    }


  ImageType::Pointer weightImage = NULL;

  if( weightImageName != "" )
    {
    typedef itk::ImageFileReader<ImageType> ReaderType;
    ReaderType::Pointer weightreader = ReaderType::New();
    weightreader->SetFileName( weightImageName.c_str() );
    weightreader->Update();
    weightImage = weightreader->GetOutput();
    }

  /**
   * convergence opions
   */
  if( numberOfIterations.size() > 1 && numberOfIterations[0] )
    {
    CorrecterType::VariableSizeArrayType
    maximumNumberOfIterations( numberOfIterations.size() );
    for( unsigned d = 0; d < numberOfIterations.size(); d++ )
      {
      maximumNumberOfIterations[d] = numberOfIterations[d];
      }
    correcter->SetMaximumNumberOfIterations( maximumNumberOfIterations );

    CorrecterType::ArrayType numberOfFittingLevels;
    numberOfFittingLevels.Fill( numberOfIterations.size() );
    correcter->SetNumberOfFittingLevels( numberOfFittingLevels );
    }

  if( convergenceThreshold )
    {
    correcter->SetConvergenceThreshold( convergenceThreshold );
    }

  /**
   * B-spline opions -- we place his here o ake care of he case where
   * he user wans o specify hings in erms of he spline disance.
   */

  ImageType::IndexType inputImageIndex =
    inputImage->GetLargestPossibleRegion().GetIndex();
  ImageType::SizeType inputImageSize =
    inputImage->GetLargestPossibleRegion().GetSize();
  ImageType::IndexType maskImageIndex =
    maskImage->GetLargestPossibleRegion().GetIndex();

  ImageType::PointType newOrigin = inputImage->GetOrigin();

  if( bsplineOrder )
    {
    correcter->SetSplineOrder(bsplineOrder);
    }

  CorrecterType::ArrayType numberOfControlPoints;
  if( splineDistance )
    {

    unsigned long lowerBound[ImageDimension];
    unsigned long upperBound[ImageDimension];
    for( unsigned  d = 0; d < 3; d++ )
      {
      float domain = static_cast<RealType>( inputImage->
                                            GetLargestPossibleRegion().GetSize()[d] - 1 ) * inputImage->GetSpacing()[d];
      unsigned int  numberOfSpans = static_cast<unsigned int>( vcl_ceil( domain / splineDistance ) );
      unsigned long extraPadding = static_cast<unsigned long>( ( numberOfSpans
                                                                 * splineDistance
                                                                 - domain ) / inputImage->GetSpacing()[d] + 0.5 );
      lowerBound[d] = static_cast<unsigned long>( 0.5 * extraPadding );
      upperBound[d] = extraPadding - lowerBound[d];
      newOrigin[d] -= ( static_cast<RealType>( lowerBound[d] )
                        * inputImage->GetSpacing()[d] );
      numberOfControlPoints[d] = numberOfSpans + correcter->GetSplineOrder();
      }

    typedef itk::ConstantPadImageFilter<ImageType, ImageType> PadderType;
    PadderType::Pointer padder = PadderType::New();
    padder->SetInput( inputImage );
    padder->SetPadLowerBound( lowerBound );
    padder->SetPadUpperBound( upperBound );
    padder->SetConstant( 0 );
    padder->Update();
    inputImage = padder->GetOutput();

    typedef itk::ConstantPadImageFilter<MaskImageType, MaskImageType> MaskPadderType;
    MaskPadderType::Pointer maskPadder = MaskPadderType::New();
    maskPadder->SetInput( maskImage );
    maskPadder->SetPadLowerBound( lowerBound );
    maskPadder->SetPadUpperBound( upperBound );
    maskPadder->SetConstant( 0 );
    maskPadder->Update();
    maskImage = maskPadder->GetOutput();

    if( weightImage )
      {
      PadderType::Pointer weightPadder = PadderType::New();
      weightPadder->SetInput( weightImage );
      weightPadder->SetPadLowerBound( lowerBound );
      weightPadder->SetPadUpperBound( upperBound );
      weightPadder->SetConstant( 0 );
      weightPadder->Update();
      weightImage = weightPadder->GetOutput();
      }
    correcter->SetNumberOfControlPoints( numberOfControlPoints );
    }
  else if( initialMeshResolution.size() == 3 )
    {
    for( unsigned d = 0; d < 3; d++ )
      {
      numberOfControlPoints[d] = static_cast<unsigned int>( initialMeshResolution[d] )
        + correcter->GetSplineOrder();
      }
    correcter->SetNumberOfControlPoints( numberOfControlPoints );
    }

  typedef itk::ShrinkImageFilter<ImageType, ImageType> ShrinkerType;
  ShrinkerType::Pointer shrinker = ShrinkerType::New();
  shrinker->SetInput( inputImage );
  shrinker->SetShrinkFactors( 1 );

  typedef itk::ShrinkImageFilter<MaskImageType, MaskImageType> MaskShrinkerType;
  MaskShrinkerType::Pointer maskshrinker = MaskShrinkerType::New();
  maskshrinker->SetInput( maskImage );
  maskshrinker->SetShrinkFactors( 1 );

  ShrinkerType::ShrinkFactorsType shrinkFactors(shrinkFactor);
  if (shrinkFactorMM != 0) {
    std::cout << "Found shrinkFactorMM: " << shrinkFactorMM << std::endl;
    shrinkFactors[0] = MAX(vcl_floor(shrinkFactorMM/inputImage->GetSpacing()[0]), 1);
    shrinkFactors[1] = MAX(vcl_floor(shrinkFactorMM/inputImage->GetSpacing()[1]), 1);
    shrinkFactors[2] = MAX(vcl_floor(shrinkFactorMM/inputImage->GetSpacing()[2]), 1);
  }
  std::cout << "Final shrink factors: " << shrinkFactors << std::endl;
  shrinker->SetShrinkFactors( shrinkFactors );
  maskshrinker->SetShrinkFactors( shrinkFactors );
  shrinker->Update();
  maskshrinker->Update();

  itk::TimeProbe timer;
  timer.Start();

  correcter->SetInput( shrinker->GetOutput() );
  correcter->SetMaskImage( maskshrinker->GetOutput() );
  if( weightImage )
    {
    typedef itk::ShrinkImageFilter<ImageType, ImageType> WeightShrinkerType;
    WeightShrinkerType::Pointer weightshrinker = WeightShrinkerType::New();
    weightshrinker->SetInput( weightImage );
    weightshrinker->SetShrinkFactors( 1 );
    weightshrinker->SetShrinkFactors( shrinkFactor );
    weightshrinker->Update();
    correcter->SetConfidenceImage( weightshrinker->GetOutput() );
    }

  typedef CommandIterationUpdate<CorrecterType> CommandType;
  CommandType::Pointer observer = CommandType::New();
  correcter->AddObserver( itk::IterationEvent(), observer );

  /**
   * histogram sharpening options
   */
  if( bfFWHM )
    {
    correcter->SetBiasFieldFullWidthAtHalfMaximum( bfFWHM );
    }
  if( wienerFilterNoise )
    {
    correcter->SetWienerFilterNoise( wienerFilterNoise );
    }
  if( nHistogramBins )
    {
    correcter->SetNumberOfHistogramBins( nHistogramBins );
    }

  try
    {
    itk::PluginFilterWatcher watchN4(correcter, "N4 Bias field correction", CLPProcessInformation, 1.0 / 1.0, 0.0);
    correcter->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }
  catch( ... )
    {
    std::cerr << "Unknown Exception caught." << std::endl;
    return EXIT_FAILURE;
    }

  correcter->Print( std::cout, 3 );

  timer.Stop();
  std::cout << "Elapsed ime: " << timer.GetMean() << std::endl;

  /**
   * ouput
   */
  if( outputImageName != "" )
    {
    /**
     * Reconsruct the bias field at full image resoluion.  Divide
     * the original input image by the bias field to get the final
     * corrected image.
     */
    typedef itk::BSplineControlPointImageFilter<
      CorrecterType::BiasFieldControlPointLatticeType,
      CorrecterType::ScalarImageType> BSplinerType;
    BSplinerType::Pointer bspliner = BSplinerType::New();
    bspliner->SetInput( correcter->GetLogBiasFieldControlPointLattice() );
    bspliner->SetSplineOrder( correcter->GetSplineOrder() );
    bspliner->SetSize( inputImage->GetLargestPossibleRegion().GetSize() );
    bspliner->SetOrigin( newOrigin );
    bspliner->SetDirection( inputImage->GetDirection() );
    bspliner->SetSpacing( inputImage->GetSpacing() );
    bspliner->Update();

    ImageType::Pointer logField = ImageType::New();
    logField->SetOrigin( inputImage->GetOrigin() );
    logField->SetSpacing( inputImage->GetSpacing() );
    logField->SetRegions( inputImage->GetLargestPossibleRegion() );
    logField->SetDirection( inputImage->GetDirection() );
    logField->Allocate();

    itk::ImageRegionIterator<CorrecterType::ScalarImageType> IB(
      bspliner->GetOutput(),
      bspliner->GetOutput()->GetLargestPossibleRegion() );
    itk::ImageRegionIterator<ImageType> IF( logField,
                                            logField->GetLargestPossibleRegion() );
    for( IB.GoToBegin(), IF.GoToBegin(); !IB.IsAtEnd(); ++IB, ++IF )
      {
      IF.Set( IB.Get()[0] );
      }

    typedef itk::ExpImageFilter<ImageType, ImageType> ExpFilterType;
    ExpFilterType::Pointer expFilter = ExpFilterType::New();
    expFilter->SetInput( logField );
    expFilter->Update();

    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DividerType;
    DividerType::Pointer divider = DividerType::New();
    divider->SetInput1( inputImage );
    divider->SetInput2( expFilter->GetOutput() );
    divider->Update();

    ImageType::RegionType inputRegion;
    inputRegion.SetIndex( inputImageIndex );
    inputRegion.SetSize( inputImageSize );

    typedef itk::ExtractImageFilter<ImageType, ImageType> CropperType;
    CropperType::Pointer cropper = CropperType::New();
    cropper->SetInput( divider->GetOutput() );
    cropper->SetExtractionRegion( inputRegion );
#if ITK_VERSION_MAJOR >= 4
    cropper->SetDirectionCollapseToSubmatrix();
#endif
    cropper->Update();

    CropperType::Pointer biasFieldCropper = CropperType::New();
    biasFieldCropper->SetInput( expFilter->GetOutput() );
    biasFieldCropper->SetExtractionRegion( inputRegion );
#if ITK_VERSION_MAJOR >= 4
    biasFieldCropper->SetDirectionCollapseToSubmatrix();
#endif

    biasFieldCropper->Update();

    if( outputBiasFieldName != "" )
      {
      typedef itk::ImageFileWriter<ImageType> WriterType;
      WriterType::Pointer writer = WriterType::New();
      writer->SetFileName( outputBiasFieldName.c_str() );
      writer->SetInput( biasFieldCropper->GetOutput() );
      writer->SetUseCompression(1);
      writer->Update();
      }

    try
      {

      itk::ImageIOBase::IOPixelType     pixelType;
      itk::ImageIOBase::IOComponentType componentType;

      itk::GetImageType(inputImageName, pixelType, componentType);

      // This filter handles all types on input, but only produces
      // signed types
      const char *fname = outputImageName.c_str();

      switch( componentType )
        {
        case itk::ImageIOBase::UCHAR:
          return SaveIt( cropper->GetOutput(), fname, static_cast<unsigned char>(0) );
          break;
        case itk::ImageIOBase::CHAR:
          return SaveIt( cropper->GetOutput(), fname, static_cast<char>(0) );
          break;
        case itk::ImageIOBase::USHORT:
          return SaveIt( cropper->GetOutput(), fname, static_cast<unsigned short>(0) );
          break;
        case itk::ImageIOBase::SHORT:
          return SaveIt( cropper->GetOutput(), fname, static_cast<short>(0) );
          break;
        case itk::ImageIOBase::UINT:
          return SaveIt( cropper->GetOutput(), fname, static_cast<unsigned int>(0) );
          break;
        case itk::ImageIOBase::INT:
          return SaveIt( cropper->GetOutput(), fname, static_cast<int>(0) );
          break;
        case itk::ImageIOBase::ULONG:
          return SaveIt( cropper->GetOutput(), fname, static_cast<unsigned long>(0) );
          break;
        case itk::ImageIOBase::LONG:
          return SaveIt( cropper->GetOutput(), fname, static_cast<long>(0) );
          break;
        case itk::ImageIOBase::FLOAT:
          return SaveIt( cropper->GetOutput(), fname, static_cast<float>(0) );
          break;
        case itk::ImageIOBase::DOUBLE:
          return SaveIt( cropper->GetOutput(), fname, static_cast<double>(0) );
          break;
        case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
          std::cerr << "Cannot saved the result using the requested pixel type" << std::endl;
          return EXIT_FAILURE;
        default:
          std::cout << "unknown component type" << std::endl;
          break;
        }
      }
    catch( itk::ExceptionObject & e )
      {
      std::cerr << "Failed to save the data: " << e << std::endl;
      return EXIT_FAILURE;
      }

    }

  return EXIT_SUCCESS;
}
