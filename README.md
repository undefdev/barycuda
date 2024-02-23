# Wasserstein Image Barycenter Calculation

## Overview
This Julia package provides functionalities for calculating Wasserstein barycenters of images. The barycenter of images is computed using the concept of Wasserstein distances, optimized through the Sinkhorn algorithm and CUDA for efficient GPU computation.

## Dependencies
- CuArrays
- LinearAlgebra
- Images
- FileIO

## Functions
- `imageBarycenter(images, t, sharpen=false)`: Computes the barycenter of two images.
- `imageBarycenters(images, step=0.05, sharpen=false)`: Computes a series of barycenters between a pair of images.
- `saveImages(folderName, images)`: Saves images to a specified folder.

## Usage
1. Prepare your images for processing.
2. Choose the weighting for barycenter computation or use the default step for a series of barycenters.
3. Call `imageBarycenter` or `imageBarycenters` to process the images.
4. Save the results using `saveImages`.

## Notes
- Ensure CUDA-compatible hardware for optimal performance.
- The package focuses on grayscale images. Preprocessing may be required for color images.

## Example
```julia
# Example usage
images = [load("image1.png"), load("image2.png")]
barycenter = imageBarycenter(images, 0.5)
saveImages("output", [barycenter])
```

---

This readme provides a basic overview and example usage of the functionalities. The actual usage might vary depending on specific requirements and setup.
