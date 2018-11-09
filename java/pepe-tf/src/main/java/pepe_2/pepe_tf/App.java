package pepe_2.pepe_tf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.DecodeCSV;
import org.tensorflow.op.core.ShuffleAndRepeatDataset;
public class App
{
	public static void main(String[] args) {
		byte[] graphDef = readAllBytesOrExit(Paths.get("/home/jdt/Desktop/entregaFinal/out", "frozen_grafopp.bytes"));
		float[][] cosos = new float[39][22];
		for (int i=0; i<cosos.length;i++) {
			cosos[i] = new float[] {0,76,3,0,0,2,20,500,5,0.4f,37,3,110,50,80,90,30,250,5,110,50,80};
		}
		
		List<String> labels = new LinkedList<String>();
		labels.add("0");
		labels.add("1");
		
	    try (Tensor<Float> data = Tensor.create(cosos, Float.class)) {
	        float[] labelProbabilities = executeInceptionGraph(graphDef, data);
	        System.out.println(labelProbabilities[0]);
	  }
	}

	private static byte[] readAllBytesOrExit(Path path) {
	    try {
	      return Files.readAllBytes(path);
	    } catch (IOException e) {
	      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
	      System.exit(1);
	    }
	    return null;
	}

	  private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] datos) {
	    try (Graph g = new Graph()) {
	      GraphBuilder b = new GraphBuilder(g);
	      final int H = 22;
	      final int W = 100;
	      final float mean = 117f;
	      final float scale = 1f;

	      final Output<String> input = b.constant("input", datos);
	      final Output<Float> output =
	          b.div(
	              b.sub(
	                  b.resizeBilinear(
	                      b.expandDims(
	                          b.cast(b.decodeJpeg(input, 3), Float.class),
	                          b.constant("make_batch", 0)),
	                      b.constant("size", new int[] {H, W})),
	                  b.constant("mean", mean)),
	              b.constant("scale", scale));
	      try (Session s = new Session(g)) {
	        // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
	        return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
	      }
	    }
	}


	  private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
	    try (Graph g = new Graph()) {
	      g.importGraphDef(graphDef);
	      try (Session s = new Session(g);
	          // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
	          Tensor<Float> result =
	              s.runner().feed("input_node", image).fetch("output_node").run().get(0).expect(Float.class)) {
	        final long[] rshape = result.shape();
	        if (result.numDimensions() != 2 || rshape[0] != 1) {
	          throw new RuntimeException(
	              String.format(
	                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
	                  Arrays.toString(rshape)));
	        }
	        int nlabels = (int) rshape[1];
	        return result.copyTo(new float[1][nlabels])[0];
	      }
	    }
	}
}

