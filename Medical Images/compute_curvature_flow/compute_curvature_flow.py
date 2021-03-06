import itk
import sys

if len(sys.argv) != 5:
    print("Usage: " + sys.argv[0] + " <inputImage> <outputImage> "
          "<numberOfIterations> <timeStep>")
    sys.exit(1)

inputImage = sys.argv[1]
outputImage = sys.argv[2]
numberOfIterations = int(sys.argv[3])
timeStep = float(sys.argv[4])

InputPixelType = itk.F
OutputPixelType = itk.UC
Dimension = 2

InputImageType = itk.Image[InputPixelType, Dimension]
OutputImageType = itk.Image[OutputPixelType, Dimension]

ReaderType = itk.ImageFileReader[InputImageType]
reader = ReaderType.New()
reader.SetFileName(inputImage)

FilterType = itk.CurvatureFlowImageFilter[
    InputImageType, InputImageType]
curvatureFlowFilter = FilterType.New()

curvatureFlowFilter.SetInput(reader.GetOutput())
curvatureFlowFilter.SetNumberOfIterations(numberOfIterations)
curvatureFlowFilter.SetTimeStep(timeStep)

RescaleFilterType = itk.RescaleIntensityImageFilter[
    InputImageType, OutputImageType]
rescaler = RescaleFilterType.New()
rescaler.SetInput(curvatureFlowFilter.GetOutput())

outputPixelTypeMinimum = itk.NumericTraits[OutputPixelType].min()
outputPixelTypeMaximum = itk.NumericTraits[OutputPixelType].max()

rescaler.SetOutputMinimum(outputPixelTypeMinimum)
rescaler.SetOutputMaximum(outputPixelTypeMaximum)

WriterType = itk.ImageFileWriter[OutputImageType]
writer = WriterType.New()
writer.SetFileName(outputImage)
writer.SetInput(rescaler.GetOutput())

writer.Update()