import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class Ccomparator implements Comparable<Ccomparator>{

	private AbstractClassifier classifier;
	private Evaluation e;
	private Instances data;
	
	public Ccomparator(AbstractClassifier classifier, Instances data) throws Exception {
		super();
		System.out.println(classifier.getClass().getSimpleName() + String.join(" ", classifier.getOptions()));
		this.classifier = classifier;
		e = new Evaluation(data);
		e.crossValidateModel(classifier, data, 10, new Random(1));
		this.data = data;
	}
	
	
	public AbstractClassifier getClassifier() {
		return classifier;
	}
	public void setClassifier(AbstractClassifier classifier) {
		this.classifier = classifier;
	}

	public Evaluation getE() {
		return e;
	}
	public void setE(Evaluation e) {
		this.e = e;
	}
	
	
	@Override
	public int compareTo(Ccomparator ccomparatorB) {
		return Double.compare(e.pctCorrect(), ccomparatorB.getE().pctCorrect());
	}

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		
		sb.append(classifier.getClass().getSimpleName() + " ");
		sb.append("(" +  String.format("%.3f", e.pctCorrect()) + ") ");
		
		sb.append(String.join(" ", classifier.getOptions()));
		sb.append('\n');
		
		for (int i = 0; i < e.confusionMatrix().length; i++) {
			for (int j = 0; j < e.confusionMatrix().length; j++) {
				sb.append(e.confusionMatrix()[i][j] + " ");
			}
			sb.append("\n");
		}
		sb.append("\n");
		
		sb.append("Correctas " + String.format("%.3f", e.pctCorrect()) + "\n");
		sb.append("Incorrectas " + String.format("%.3f", e.pctIncorrect()) + "\n");
		sb.append("Error Abs Medio " + String.format("%.3f", e.meanAbsoluteError()) + "\n");
		sb.append("Error Cuadratico Medio " + String.format("%.3f", e.rootMeanSquaredError()) + "\n");
		try {
			sb.append("Error Abs Relativo " + String.format("%.3f", e.relativeAbsoluteError()) + "\n");
			sb.append("Error Cuadratico relativo " + String.format("%.3f", e.rootRelativeSquaredError()) + "\n");
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		sb.append("Kappa " + String.format("%.3f", e.kappa()) + "\n");
		
		for (int i=0; i<data.numClasses(); i++) {
			sb.append("Clase " + data.classAttribute().value(i) + "\n");
			sb.append("precision: " + String.format("%.3f", e.precision(i)) + "\n");
			sb.append("recall: " + String.format("%.3f", e.recall(i)) + "\n");
			sb.append("roc: " + String.format("%.3f", e.areaUnderROC(i)) + "\n");
		}
		sb.append("\n");
		
		return sb.toString();
	}
}