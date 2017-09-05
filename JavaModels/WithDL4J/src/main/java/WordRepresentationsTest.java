import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
//import org.nd4j.linalg.activations.Activation;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.*;
import java.nio.charset.Charset;

/**
 * Решение задачи https://github.com/Koziev/WordRepresentations с помошью глубокой сетки,
 * созданной средствами Deeplearning4j.
 *
 * Предполагается, что датасеты для тренировки, валидации и финальной оценки сгенерированы скриптом
 * PyModels/store_dataset_file.py и созранены в папке data как файлы Xy_train.csv, Xy_val.csv,
 * Xy_holdout.csv.
 *
 * (c) Koziev Ilya inkoziev@gmail.com
 */
public class WordRepresentationsTest {


    private static int getDatasetFeaturesCount( String dataset_path ) {
        try {
            String delimiter = "\t";

            File file = new File(dataset_path);

            InputStream fis = new FileInputStream(file);
            InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
            BufferedReader br = null;

            try
            {
             br = new BufferedReader(isr);
             String line0 = br.readLine();
             int input_size = line0.split(delimiter).length - 1;
             return input_size;
            }
            finally
            {
                br.close();
            }
        }
        catch(Exception ex) {
            System.out.println("Could not load file "+dataset_path );
            System.exit(1);
            return -1;
        }
    }


    private static DataSetIterator loadDataset( String dataset_path, int input_size, int batch_size )
    {
        try {
            System.out.format("Loading dataset from %s...", dataset_path);

            char delimiter = '\t';

            File file = new File(dataset_path);

            int real_batch_size = batch_size;
            if( real_batch_size<=0 ) {
                LineNumberReader lineNumberReader = new LineNumberReader(new FileReader(file));
                lineNumberReader.skip(Long.MAX_VALUE);
                real_batch_size = lineNumberReader.getLineNumber();
            }


            int numLinesToSkip = 0;
            RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
            recordReader.initialize(new FileSplit(file));

            int labelIndex = input_size;
            int numClasses = 2;

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,real_batch_size,labelIndex,numClasses);

            System.out.println("done");

            return iterator;
        }
        catch(Exception ex)
        {
            System.out.println("Could not load file "+dataset_path );
            System.exit(1);
            return null;
        }
    }

    private static MultiLayerNetwork createModel(int input_size)
    {
        int numOutputs = 2;
        int numHiddenNodes = input_size;
        int seed=123456;
        double learning_rate = 0.05;


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learning_rate)
                .updater(Updater.NESTEROVS)     //To configure: .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(input_size).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())

                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes/2)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())

                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes/2).nOut(numHiddenNodes/3)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())

                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes/3).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates

        return model;
    }

    public static void main(String[] args) {

        int batch_size = 100;

        String train_path = "/home/eek/polygon/WordRepresentations/data/Xy_train.csv";
        int input_size = getDatasetFeaturesCount(train_path);


        DataSetIterator train_iter = loadDataset(train_path, input_size, batch_size);

        DataSetIterator val_iter = loadDataset("/home/eek/polygon/WordRepresentations/data/Xy_val.csv",  input_size,0);
        DataSet val_ds = val_iter.next();
        INDArray x_val = val_ds.getFeatureMatrix();
        INDArray y_val = val_ds.getLabels();

        int numOutputs = 2;
        int numHiddenNodes = input_size;

        MultiLayerNetwork model = createModel(input_size);

        String checkpoint_file = "/home/eek/polygon/WordRepresentations/data/deeplearning4j.model";

        int nEpochs = 50;
        double best_val_acc = 0.0;
        int nb_no_improvements = 0;

        for ( int iter = 0; iter < nEpochs; iter++) {
            System.out.format("Start iteration #%d\n", iter);
            model.fit( train_iter );

            System.out.println("Evaluating...");

            INDArray output = model.output(x_val);
            //System.out.println(output);

            // let Evaluation prints stats how often the right output had the
            // highest value
            Evaluation eval = new Evaluation(2);
            eval.eval(y_val, output);
            System.out.println(eval.stats());

            double val_acc = eval.accuracy();
            if( val_acc>best_val_acc ) {
                best_val_acc = val_acc;
                nb_no_improvements = 0;

                System.out.format("New best val_acc=%s\n", best_val_acc);

                try {
                    File locationToSave = new File(checkpoint_file);
                    boolean saveUpdater = false;
                    ModelSerializer.writeModel(model, locationToSave, saveUpdater);
                    System.out.format("Model saved in %s\n", checkpoint_file);
                }
                catch(java.io.IOException ex) {
                    System.out.println("Error occured when saving model to file: "+ex.getMessage());
                    System.exit(-1);
                }
            }
            else
            {
                nb_no_improvements++;
                if( nb_no_improvements>10) {
                System.out.format("Early stopping occured after not improving val_acc during %d epochs\n", nb_no_improvements);
                break;
                }
            }
        }

        MultiLayerNetwork restored_model = null;
        try  {
            // Загрузим оптимальный вариант весов сетки.
            restored_model = ModelSerializer.restoreMultiLayerNetwork(checkpoint_file);
        }
        catch(java.io.IOException ex) {
            System.out.println("Error occured when saving model to file: "+ex.getMessage());
            System.exit(-1);
        }


     System.out.println("Final evaluation with holdout dataset");
     DataSetIterator holdout_iter = loadDataset("/home/eek/polygon/WordRepresentations/data/Xy_holdout.csv", input_size, 0);
     DataSet holdout_ds = holdout_iter.next();
     INDArray x_holdout = holdout_ds.getFeatureMatrix();
     INDArray y_holdout = holdout_ds.getLabels();

     INDArray output = restored_model.output(x_holdout);
     Evaluation eval = new Evaluation(2);
     eval.eval(y_holdout, output);
     System.out.println(eval.stats());

     return;
    }
}


