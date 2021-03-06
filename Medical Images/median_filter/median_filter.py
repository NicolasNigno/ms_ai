import sys
import itk

if len(sys.argv) != 4:
    print("Usage: " + sys.argv[0] + " <inputImage> <outputImage> <radius>")
    sys.exit(1)

inputImage = sys.argv[1]
outputImage = sys.argv[2]
radius = int(sys.argv[3])

PixelType = itk.UC
Dimension = 2

ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(inputImage)

medianFilter = itk.MedianImageFilter[ImageType, ImageType].New()
medianFilter.SetInput(reader.GetOutput())
medianFilter.SetRadius(radius)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName(outputImage)
writer.SetInput(medianFilter.GetOutput())

writer.Update()