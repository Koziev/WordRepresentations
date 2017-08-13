// Решение классификационной задачи, описанной тут: https://github.com/Koziev/WordRepresentations
// Файлы с датасетами в формате csv с табуляцией в качестве разделителя полей должны быть
// заранее подготовлены скриптом store_dataset_file.py
//
// Решатель - простая feed forward нейросетка на базе инструментов библиотеки
// Accord.Net (http://accord-framework.net/), устанавливаемой через NuGet.

using System;
using System.Linq;
using Accord.Neuro;
using Accord.Neuro.Learning;


namespace WithAccordNet
{
    class Program
    {
        // В этой папке должны лежать файлы, сгенерированные скриптом store_dataset_file.py (см. репозиторий).
        static string data_folder = "../../../../data";

        static double[][] LoadData(string filename)
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
            double[][] data = new double[nb_rows][];
            for (int i = 0; i < nb_rows; ++i)
            {
                data[i] = new double[nb_cols];
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
                        data[irow][icol] = double.Parse(x[icol], System.Globalization.CultureInfo.InvariantCulture);
                    }
                    irow++;
                }
            }

            return data;
        }


        static void Main(string[] args)
        {
            double learningRate = 0.1;
            double sigmoidAlphaValue = 2;

            // iterations
            int iterations = 100;

            bool useNguyenWidrow = false;

            var X_train = LoadData("X_train.csv");
            var y_train = LoadData("y_train.csv");

            var X_val = LoadData("X_val.csv");
            var y_val = LoadData("y_val.csv");

            var X_holdout = LoadData("X_holdout.csv");
            var y_holdout = LoadData("y_holdout.csv");

            int nb_samples = X_train.Length;
            int input_size = X_train[0].Length;

            // create multi-layer neural network
            ActivationNetwork ann = new ActivationNetwork(
                new BipolarSigmoidFunction(sigmoidAlphaValue),
                input_size,
                input_size, 1);

            if (useNguyenWidrow)
            {
                NguyenWidrow initializer = new NguyenWidrow(ann);
                initializer.Randomize();
            }

            // create teacher
            //LevenbergMarquardtLearning teacher = new LevenbergMarquardtLearning(ann, useRegularization);
            BackPropagationLearning teacher = new BackPropagationLearning(ann);

            // set learning rate and momentum
            teacher.LearningRate = learningRate;
            teacher.Momentum = 0.8;

            // iterations
            int iteration = 1;

            //var ranges = Matrix.Range(sourceMatrix, 0);
            //double[][] map = Matrix.Mesh(ranges[0], ranges[1], 0.05, 0.05);
            // var sw = Stopwatch.StartNew();

            bool use_batches = false;
            int batch_size = 5000;
            double[][] X_batch = new double[batch_size][];
            double[][] y_batch = new double[batch_size][];

            var rng = new Random();

            while (true)
            {
                double error = 0.0;
                if (use_batches)
                {
                    int[] sample_indeces = Enumerable.Range(0, nb_samples).OrderBy(z => rng.Next()).ToArray();
                    int nb_batch = nb_samples / batch_size;

                    int n_grad = 50;
                    Console.Write("|{0}| error=\r", new string('-', n_grad));

                    for (int ibatch = 0; ibatch < nb_batch; ++ibatch)
                    {
                        for (int i = 0; i < batch_size; ++i)
                        {
                            X_batch[i] = X_train[sample_indeces[i]];
                            y_batch[i] = y_train[sample_indeces[i]];
                        }

                        error += teacher.RunEpoch(X_batch, y_batch);

                        int ngrad1 = (int)Math.Ceiling((ibatch+1) * n_grad / (double)(nb_batch));
                        int ngrad0 = n_grad - ngrad1;
                        double cur_err = error / (batch_size*(ibatch+1));
                        Console.Write("|{0}{1}| error={2:0.####}\r", new string('#', ngrad1), new string('-', ngrad0), cur_err);
                    }

                    error /= (batch_size*nb_batch);
                }
                else
                {
                    error = teacher.RunEpoch(X_train, y_train) / nb_samples;
                }

                // Получим оценку точности на валидационном наборе
                int n_hit2 = 0;
                for (int i = 0; i < X_val.Length; ++i)
                {
                    double[] output = ann.Compute(X_val[i]);
                    double y_bool = output[0] > 0.5 ? 1.0 : 0.0;
                    n_hit2 += y_bool == y_val[i][0] ? 1 : 0;
                }

                double val_acc = n_hit2 / (double)X_val.Length;

                // TODO: тут надо бы смотреть, увеличилось ли качество сетки на валидации по сравнению с предыдущей эпохой,
                // сохранять веса сетки в случае улучшения (через deep copy ann), и делать остановку в случае, если качество
                // не растет уже 5-10 эпох.

                Console.WriteLine($"\niteration={iteration} train_mse={error} val_acc={val_acc}");

                iteration++;
                if ((iterations != 0) && (iteration > iterations))
                    break;
            }

            // Получим оценку точности на отложенном наборе
            int n_hit = 0;
            for (int i = 0; i < X_holdout.Length; ++i)
            {
                double[] output = ann.Compute(X_holdout[i]);
                double y_bool = output[0] > 0.5 ? 1.0 : 0.0;
                n_hit += y_bool == y_holdout[i][0] ? 1 : 0;
            }

            double holdout_acc = n_hit / (double)X_holdout.Length;
            Console.WriteLine($"holdout_acc={holdout_acc}");


            return;
        }
    }
}
