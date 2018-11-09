package pepe_2.pepe_tf;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

public class IrisReader {
	ArrayList<Double[]> features;
	ArrayList<Integer> species;
	
	public IrisReader(String path) {
		features = new ArrayList<Double[]>();
		species = new ArrayList<Integer>();
		
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line = reader.readLine();
			line = reader.readLine();
			while(line != null) {
				decodeLine(line);
				line = reader.readLine();
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	private void decodeLine(String line) {
		String parts[] = line.split(",");
		
		Double[] feature = new Double[4];
		
		for (int i=0; i<4; i++) {
			feature[i] = Double.parseDouble(parts[i]);
		}
		
		features.add(feature);
		species.add(Integer.parseInt(parts[parts.length-1]));
	}

	public ArrayList<Double[]> getFeatures() {
		return features;
	}

	public void setFeatures(ArrayList<Double[]> features) {
		this.features = features;
	}

	public ArrayList<Integer> getSpecies() {
		return species;
	}

	public void setSpecies(ArrayList<Integer> species) {
		this.species = species;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		Iterator<Double[]> featureIt = features.iterator();
		Iterator<Integer> speciesIt = species.iterator();
		
		Double[] feature = new Double[4];
		while(featureIt.hasNext() && speciesIt.hasNext()) {
			feature = featureIt.next();
			sb.append('(');
			for (Double f : feature) {
				sb.append(f + ", ");
			}
			sb.append("): " + speciesIt.next());
			sb.append('\n');
		}
		
		return sb.toString();
	}
	
	
}
