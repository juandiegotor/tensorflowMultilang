import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.ChebyshevDistance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.FilteredDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.MinkowskiDistance;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.kdtrees.KMeansInpiredMethod;
import weka.core.pmml.jaxbbindings.Cluster;

public class Mainn {
	
	private ArrayList<Ccomparator> listaComparators;
	private Instances data;
	private Instances clusterData;
	
	public Mainn(Instances data, Instances clusterData) {
		super();
		this.data = data;
		this.clusterData = clusterData;
		listaComparators = new ArrayList<Ccomparator>();
	}
	
	public void addSMO (double min, double max, double step) throws Exception {
		SMO smo = null;
		PolyKernel pk = null;
		
		for (double expo=min; expo<=max; expo+=step) {
			pk = new PolyKernel(data, 250007, expo, false);
			smo = new SMO();
			smo.setKernel(pk);
			listaComparators.add(new Ccomparator(smo, data));
		}
	}
	
	public void addMLP (int minH, int maxH, int stepH,
						double minL, double maxL, double stepL,
						double minM, double maxM, double stepM) throws Exception 
	{
		MultilayerPerceptron mlp = null;
		
		for (int i = minH; i<= maxH; i+=stepH) {
			for (double l=minL; l<=maxL; l+=stepL) {
				for (double m=minM; m<=maxM; m+=stepM) {
					//System.out.printf("%d %f %f\n", i, l, m);
					mlp = new MultilayerPerceptron();
					mlp.setHiddenLayers(Integer.toString(i));
					mlp.setLearningRate(l);
					mlp.setMomentum(m);
					listaComparators.add(new Ccomparator(mlp, data));
				}
			}
		}
	}
	
	public void addIBK (int min, int max, int step) throws Exception {
		IBk ibk = null;
		DistanceFunction[] df = { new ChebyshevDistance(data), new EuclideanDistance(), new FilteredDistance(),
								  new ManhattanDistance(), new MinkowskiDistance()};
		
		for (int knn=min; knn<=max; knn+=step) {
			for (int i=0; i<df.length; i++) {
				ibk = new IBk();
				ibk.setKNN(knn);
				ibk.getNearestNeighbourSearchAlgorithm().setDistanceFunction(df[i]);
				
				listaComparators.add(new Ccomparator(ibk, data));
			}
		}
	}
	
	public void addBayesNet () throws Exception {
		listaComparators.add(new Ccomparator(new BayesNet(), data));
	}
	
	public void addNaivBayes () throws Exception {
		listaComparators.add(new Ccomparator(new NaiveBayes(), data));
	}
	
	public void addJ48 () throws Exception {
		listaComparators.add(new Ccomparator(new J48(), data));
	}
	
	public void addBoost(int k, int h) throws Exception {
		AdaBoostM1 boost = null;
		
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		mlp.setHiddenLayers(Integer.toString(h));
		
		AbstractClassifier [] clasificadores = {
				new NaiveBayes(),
				new J48(),
				new IBk(k),
				mlp,
				new BayesNet()
		};
		
		for (int i=0; i<clasificadores.length; i++) {
			boost = new AdaBoostM1();
			boost.setClassifier(clasificadores[i]);
			listaComparators.add(new Ccomparator(boost, data));
		}
	}
	
	public void addBagging (int k, int h) throws Exception {
		Bagging b = null;
		
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		mlp.setHiddenLayers(Integer.toString(h));
		
		AbstractClassifier [] clasificadores = {
				new NaiveBayes(),
				new J48(),
				
				new IBk(k),
				mlp,
				new BayesNet()
		};
		
		for (int i=0; i<clasificadores.length; i++) {
			b = new Bagging();
			b.setClassifier(clasificadores[i]);
			listaComparators.add(new Ccomparator(b, data));
		}
	}
	
	public void addStacking (int k, int h) throws Exception {
		Stacking s = null;
		
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		mlp.setHiddenLayers(Integer.toString(h));
		
		MultilayerPerceptron mlp2 = new MultilayerPerceptron();
		mlp2.setHiddenLayers(Integer.toString(h));
		
		AbstractClassifier [] clasificadores = {
				new NaiveBayes(),
				new J48(),
				new IBk(k),
				mlp,
				new BayesNet()
		};
		
		AbstractClassifier [] c2 = {
				new NaiveBayes(),
				new J48(),
				new IBk(k),
				mlp2,
				//new BayesNet()
		};

		for (int i=0; i<c2.length; i++) {
			s = new Stacking();
			s.setClassifiers(clasificadores);
			s.setMetaClassifier(c2[i]);
			listaComparators.add(new Ccomparator(s, data));
		}
		
	}
	
	public void printCcomp () {
		Collections.sort(listaComparators, Collections.reverseOrder());
		Iterator<Ccomparator> it = listaComparators.iterator();
		while (it.hasNext()) {
			System.out.println(it.next());
		}
	}

	public static void main(String[] args) throws Exception {
		String dbPath = "/home/jdt/Downloads/exitosFiltrado.arff";
		DataSource source = new DataSource(dbPath);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		
		DataSource source2 = new DataSource(dbPath);
		Instances dataClust = source2.getDataSet();
		
		Mainn m = new Mainn(data, dataClust);
		m.addSMO(0.5, 5, 0.5);
		m.addMLP(1, 2, 1, 0.1, 0.4, 0.1, 0.1, 0.2, 0.1);
		m.addIBK(1, 5, 2);
		m.addNaivBayes();
		m.addJ48();
		m.addBayesNet();
		m.addBoost(3, 2);
		m.addBagging(3, 2);
		m.addStacking(3, 2);
		
		m.printCcomp();
		
		System.out.println("The End!");
	}
}