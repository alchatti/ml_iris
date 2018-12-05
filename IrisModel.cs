using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace ml_iris
{
    class IrisData
    {
        [Column("0")]
        public float SepalLength;
        [Column("1")]
        public float SepalWidth;
        [Column("2")]
        public float PetalLength;
        [Column("3")]
        public float PetalWidth;
        [Column("4", name: "Label")]
        public string type;

        //Default
        public IrisData(){}
        //For Processing CSV Line Feed
        public IrisData(string csvLine)
        {
            var items = csvLine.Split(',');
            SepalLength = float.Parse(items[0]);
            SepalWidth = float.Parse(items[1]);
            PetalLength = float.Parse(items[2]);
            PetalWidth = float.Parse(items[3]);
            type = items[4];
        }
    }
    class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
