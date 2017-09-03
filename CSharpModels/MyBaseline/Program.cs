// Решение классификационной задачи, описанной тут: https://github.com/Koziev/WordRepresentations
// Файлы с датасетами в формате csv с табуляцией в качестве разделителя полей должны быть
// заранее подготовлены скриптом store_dataset_file.py
// Используется моя простая реализация vanilla MLP с одним скрытым слоем и sigmoida активациями.
// Достигаемая точность - примерно 0.64

using System;
using System.Linq;


namespace WithMyMLP
{
    class Program
    {
        // В этой папке должны лежать файлы, сгенерированные скриптом store_dataset_file.py (см. https://github.com/Koziev/WordRepresentations/blob/master/PyModels/store_dataset_file.py).
        static string data_folder = "../../../../data";

        static float[][] LoadXData(string filename)
        {
            string path = System.IO.Path.Combine(data_folder, filename);
            int nb_cols = -1;
            int nb_rows = -1;
            using (System.IO.StreamReader rdr = new System.IO.StreamReader(path))
            {
                nb_cols = rdr.ReadLine().Split('\t').Length;
                nb_rows = 1;
                while (!rdr.EndOfStream)
                {
                    string line = rdr.ReadLine();
                    if (line == null)
                    {
                        break;
                    }
                    nb_rows++;
                }
            }

            Console.WriteLine($"Load matrix {nb_rows}x{nb_cols} from {path}");
            float[][] data = new float[nb_rows][];
            for (int i = 0; i < nb_rows; ++i)
            {
                data[i] = new float[nb_cols];
            }


            using (System.IO.StreamReader rdr = new System.IO.StreamReader(path))
            {
                int irow = 0;
                while (!rdr.EndOfStream)
                {
                    string line = rdr.ReadLine();
                    if (line == null)
                    {
                        break;
                    }

                    var x = line.Split('\t');
                    for (int icol = 0; icol < x.Length; ++icol)
                    {
                        data[irow][icol] = float.Parse(x[icol], System.Globalization.CultureInfo.InvariantCulture);
                    }
                    irow++;
                }
            }

            return data;
        }

        static float[][] LoadYData(string filename)
        {
            string path = System.IO.Path.Combine(data_folder, filename);
            int nb_rows = -1;
            using (System.IO.StreamReader rdr = new System.IO.StreamReader(path))
            {
                nb_rows = 1;
                while (!rdr.EndOfStream)
                {
                    string line = rdr.ReadLine();
                    if (line == null)
                    {
                        break;
                    }
                    nb_rows++;
                }
            }

            Console.WriteLine($"Load matrix {nb_rows}x{1} from {path}");
            float[][] data = new float[nb_rows][];
            for( int i=0; i<nb_rows; ++i)
            {
                data[i] = new float[1];
            }

            using (System.IO.StreamReader rdr = new System.IO.StreamReader(path))
            {
                int irow = 0;
                while (!rdr.EndOfStream)
                {
                    string line = rdr.ReadLine();
                    if (line == null)
                    {
                        break;
                    }

                    data[irow][0] = float.Parse(line.Trim(), System.Globalization.CultureInfo.InvariantCulture);
                    irow++;
                }
            }

            return data;
        }

        static void Main(string[] args)
        {
            var X_train = LoadXData("X_train.csv");
            var y_train = LoadYData("y_train.csv");

            var X_val = LoadXData("X_val.csv");
            var y_val = LoadYData("y_val.csv");

            var X_holdout = LoadXData("X_holdout.csv");
            var y_holdout = LoadYData("y_holdout.csv");

            int nb_samples = X_train.Length;
            int input_size = X_train[0].Length;

            FeedForwardNetworks.FeedForwardNetwork2 mlp = new FeedForwardNetworks.FeedForwardNetwork2(false, input_size, input_size, 1);
            mlp.LearningRate = 0.01f;
            mlp.BatchSize = 20;

            Random rnd = new Random();

            for (int iter = 0; iter < 100; ++iter)
            {
                Console.Write($"Epoch {iter}");

                int nb_batches = nb_samples / mlp.BatchSize;
                int[] indeces = Enumerable.Range(0, nb_samples).OrderBy(z => rnd.Next()).ToArray();
                for( int ibatch=0, ii=0; ibatch<nb_batches; ++ibatch)
                {
                    mlp.BeginBatch();
                    for( int idata=0; idata< mlp.BatchSize; ++idata, ++ii)
                    {
                        int irow = indeces[ii];
                        mlp.ForwardPropagation(X_train[irow]);
                        mlp.BackwardPropagation(X_train[irow], y_train[irow]);
                    }
                    mlp.EndBatch();
                }


                float loss = 0;
                int hits = 0;
                for( int irow=0; irow<X_val.Length; ++irow)
                {
                    mlp.ForwardPropagation(X_val[irow]);
                    float delta = y_val[irow][0] - mlp.GetOutput(0);
                    loss += delta * delta;

                    float z = mlp.GetOutput(0) > 0.5f ? 1 : 0;
                    if( z==y_val[irow][0] )
                    {
                        hits++;
                    }
                }

                loss /= X_val.Length;
                float acc = hits / (float)X_val.Length;
                Console.WriteLine($" val_loss={loss} val_accuracy={acc}");
            }

            return;
        }
    }
}
