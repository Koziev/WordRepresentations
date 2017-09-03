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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.api.util.ClassPathResource;

import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;

/**
 * This basic example shows how to manually create a DataSet and train it to an
 * basic Network.
 * <p>
 * The network consists in 2 input-neurons, 1 hidden-layer with 4
 * hidden-neurons, and 2 output-neurons.
 * <p>
 * I choose 2 output neurons, (the first fires for false, the second fires for
 * true) because the Evaluation class needs one neuron per classification.
 *
 * @author Peter Großmann
 */
public class WordRepresentationsTest {

    private static DataSetIterator loadDataset( String dataset_path, int input_size, int batch_size )
    {
        try {
            System.out.format("Loading dataset from %s...", dataset_path);

            File file = new File(dataset_path);

            int real_batch_size = batch_size;
            if( real_batch_size<=0 ) {
                LineNumberReader lineNumberReader = new LineNumberReader(new FileReader(file));
                lineNumberReader.skip(Long.MAX_VALUE);
                real_batch_size = lineNumberReader.getLineNumber();
            }


            int numLinesToSkip = 0;
            char delimiter = '\t';
            RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
            //recordReader.initialize(new FileSplit(new ClassPathResource(dataset_path).getFile()));
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

    public static void main(String[] args) {

        int input_size = 96;
        int batch_size = 100;
        int seed=123456;
        double learning_rate = 0.1;

        int numOutputs = 2;
        int numHiddenNodes = input_size;

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
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates


        DataSetIterator train_iter = loadDataset("/home/eek/polygon/WordRepresentations/data/Xy_train.csv", input_size, batch_size);

        DataSetIterator val_iter = loadDataset("/home/eek/polygon/WordRepresentations/data/Xy_val.csv", input_size, 0);
        DataSet val_ds = val_iter.next();
        INDArray x_val = val_ds.getFeatureMatrix();
        INDArray y_val = val_ds.getLabels();

        int nEpochs = 30;

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
        }

     System.out.println("Final evaluation with holdout dataset");
     DataSetIterator holdout_iter = loadDataset("/home/eek/polygon/WordRepresentations/data/Xy_holdout.csv", input_size, 0);
     DataSet holdout_ds = holdout_iter.next();
     INDArray x_holdout = holdout_ds.getFeatureMatrix();
     INDArray y_holdout = holdout_ds.getLabels();

     INDArray output = model.output(x_holdout);
     Evaluation eval = new Evaluation(2);
     eval.eval(y_holdout, output);
     System.out.println(eval.stats());

     return;
    }
}


