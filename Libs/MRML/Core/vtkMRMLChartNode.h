/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLChartNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
///  vtkMRMLChartNode - MRML node for referencing a collection of data
///  to plot
/// 
//

#ifndef __vtkMRMLChartNode_h
#define __vtkMRMLChartNode_h

#include "vtkMRMLNode.h"

class vtkDataObject;
class vtkStringArray;
class DoubleArrayIDMap;

#include <string>

class VTK_MRML_EXPORT vtkMRMLChartNode : public vtkMRMLNode
{
 public:
  //----------------------------------------------------------------
  /// Constants
  //----------------------------------------------------------------



  //----------------------------------------------------------------
  /// Standard methods for MRML nodes
  //----------------------------------------------------------------

  static vtkMRMLChartNode *New();
  vtkTypeMacro(vtkMRMLChartNode,vtkMRMLNode);
  
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLNode* CreateNodeInstance();

  /// 
  /// Set node attributes
  virtual void ReadXMLAttributes( const char** atts);

  /// 
  /// Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  /// 
  /// Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);

  /// 
  /// Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName()
    {return "Chart";};

  /// 
  /// Method to propagate events generated in mrml
  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData );
  
  //----------------------------------------------------------------
  /// Access methods
  //----------------------------------------------------------------

  ///
  /// Add a double array to the chart. Parameter "name" is used for
  /// referencing the array when setting properties for plotting the
  /// array or for removing the array from the chart.
  void AddArray(const char *name, const char * id);
  
  ///
  /// Remove a double array from the chart
  void RemoveArray(const char *name);

  ///
  /// Remove all the arrays
  void ClearArrays();
  
  ///
  /// Set a property for a specific array to control how it will
  /// appear in the chart. Properties can control lines, points,
  /// colors, etc.
  void SetProperty(const char *name, const char *property, const char *value);

  ///
  /// Get the array id referenced by a particular name
  const char *GetArray(const char *name);

  ///
  /// Get the list of array names
  vtkStringArray* GetArrayNames();

  ///
  /// Get the list of array ids
  vtkStringArray* GetArrays();

  //----------------------------------------------------------------
  /// Constructor and destroctor
  //----------------------------------------------------------------
 protected:
  vtkMRMLChartNode();
  ~vtkMRMLChartNode();
  vtkMRMLChartNode(const vtkMRMLChartNode&);
  void operator=(const vtkMRMLChartNode&);


 protected:
  //----------------------------------------------------------------
  /// Data
  //----------------------------------------------------------------
  DoubleArrayIDMap *DoubleArrayIDs;
  vtkStringArray *Arrays;        // only valid when GetArrays() is called
  vtkStringArray *ArrayNames;    // only valid when GetArrayNames() is called
};



#endif
