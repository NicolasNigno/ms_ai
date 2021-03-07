#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkGradientMagnitudeImageFilter.h"

int
main(int argc, char * argv[])
{
  // Verify command line arguments
  if (argc != 3)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " <InputImage> <OutputImage" << std::endl;
    return EXIT_FAILURE;
  }

  const char * inputFileName = argv[1];
  const char * outputFileName = argv[2];

  constexpr unsigned int Dimension = 2;

  using InputPixelType = unsigned char;
  using InputImageType = itk::Image<InputPixelType, Dimension>;

  using OutputPixelType = unsigned char;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Create and setup a reader
  using ReaderType = itk::ImageFileReader<InputImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFileName);

  // Create and setup a gradient filter
  using FilterType = itk::GradientMagnitudeImageFilter<InputImageType, OutputImageType>;

  FilterType::Pointer gradientFilter = FilterType::New();
  gradientFilter->SetInput(reader->GetOutput());

  using WriterType = itk::ImageFileWriter<OutputImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(outputFileName);
  writer->SetInput(gradientFilter->GetOutput());

  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject & error)
  {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}