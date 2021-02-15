#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <iostream>

// class GridType
// {
// public:
//   GridType()
//   {
//     score_ = 0.0f;
//     num_ = 0.0f;
//   }
//   float score_;
//   float num_;
// };

class Grid3D
{
public:
  Grid3D();
  Grid3D(const unsigned int dimX, const unsigned int dimY, const unsigned int dimZ);
  ~Grid3D();
  
  void allocate(const unsigned int dimX, const unsigned int dimY, const unsigned int dimZ);
  void deallocate();
  
  void printInfo() const;
  
  // The 3D data is stored in a 1D array that is ordered such that
  // volume[x + dimX*(y + dimY*z)] = data value at (x,y,z).
  
  // Accessing elements of the grid:
  
  // At integer location
  inline float getGridValueAt(const unsigned int ix, const unsigned int iy, const unsigned int iz) const
  {
    return data_array_.at(ix + size_[0]*(iy + size_[1]*iz));
  }

  inline float GridValueRatio(const float v);

  // At floating point location within a Z slice. Bilinear interpolation or voting.
  inline void accumulateGridValueAt(const float x_f, const float y_f, float* grid);

  void collapseMaxZSlice(cv::Mat* max_val, cv::Mat* max_pos) const;
  
  // Statistics
  double computeMeanSquare() const;
  
  void resetGrid();
  
  // Output
  int writeGridNpy(const char szFilename[]) const;
  
  void getDimensions(int* dimX, int* dimY, int* dimZ) const
  {
    *dimX = size_[0];
    *dimY = size_[1];
    *dimZ = size_[2];
  }

  float* getPointerToSlice(int layer)
  {
    return &data_array_.data()[layer * size_[0] * size_[1]];
  }

private:
  std::vector<float> data_array_;
  unsigned int numCells_;
  unsigned int size_[3];
  
};


inline float Grid3D::GridValueRatio(const float v)
{
  const float r[10] = {1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8};
  // const float r[10] = {1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 5.0, 5.5, 6.0, 6.5};
  return (int(v / 30) >= 10 ? r[9] : r[int(v / 30)]);
}

// Function implementation
// Bilinear voting within a Z-slice, if point (x,y) is given as float
inline void Grid3D::accumulateGridValueAt(const float x_f, const float y_f, float* grid)
{
  if (x_f >= 0.f && y_f >= 0.f)
  {
    const int x = x_f, y = y_f;
    if (x+1 < size_[0] &&
        y+1 < size_[1])
    {
      float* g = grid + x + y * size_[0];
      const float fx = x_f - x,
          fy = y_f - y,
          fx1 = 1.f - fx,
          fy1 = 1.f - fy;

      g[0] += GridValueRatio(g[0]) * fx1*fy1;
      g[1] += GridValueRatio(g[1]) * fx*fy1;
      g[size_[0]]   += GridValueRatio(g[size_[0]]) * fx1*fy;
      g[size_[0]+1] += GridValueRatio(g[size_[0]+1]) * fx*fy;
    }
  }
}
