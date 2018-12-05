using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using System;
using System.IO;
using System.Linq;

namespace ml_iris
{
    class Program
    {
        public static string trainingFile = @"Data/iris_train.csv";
        public static string TestFile = @"Data/iris_test.csv";

        static void Main(string[] args)
        {
            Console.WriteLine("***** Welcome *****\n-Checking Files:");
            Console.WriteLine($" - {trainingFile} : {File.Exists(trainingFile)}");
            Console.WriteLine($" - {TestFile} : {File.Exists(TestFile)}");

            if (File.Exists(trainingFile) && File.Exists(TestFile))
            {
                Console.WriteLine("Learning .....");
                var mlContext = new MLContext();
                var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
                {
                    Separator = ",",
                    HasHeader = true,
                    Column = new[]
                    {
                    new TextLoader.Column("SepalLength", DataKind.R4,0),
                    new TextLoader.Column("SepalWidth", DataKind.R4,1),
                    new TextLoader.Column("PetalLength", DataKind.R4,2),
                    new TextLoader.Column("PetalWidth", DataKind.R4,3),
                    new TextLoader.Column("Label", DataKind.Text,4)
                }
                });
                IDataView trainingData = reader.Read(new MultiFileSource(trainingFile));

                var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                               .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                               .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features"))
                               .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                var model = pipeline.Fit(trainingData);

                var predictionFunction = model.MakePredictionFunction<IrisData, IrisPrediction>(mlContext);

                Console.WriteLine("Test Data Results ....\n");
                //Load the Test Data from CSV file
                var testData = File.ReadLines(TestFile).Select(line => new IrisData(line)).ToList();

                //Shuffel the List -- Not needed Just for Fun
                testData = testData.OrderBy(a => Guid.NewGuid()).ToList();

                foreach (var iris in testData)
                {
                    Console.WriteLine($"Predicted : {predictionFunction.Predict(iris).PredictedLabels} \nActual    : {iris.type}\n=================================");
                }
                Console.WriteLine("Completed .....");
            }
            else
            {
                Console.WriteLine("Error Missing Required File(s) !");
            }
           
            Console.ReadKey();
                    
        }
    }
}
