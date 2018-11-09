package pepe_2.pepe_tf;

import java.util.Random;

import org.tensorflow.Graph;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.DecodeCSV;
import org.tensorflow.op.core.ShuffleAndRepeatDataset;
public class App
{
	
    public static void main( String[] args ) {
    	IrisReader ir = new IrisReader("/home/jdt/Desktop/iris.csv");
    	ShuffleAndRepeatDataset()
    }
}