package br.koch.marcio;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class KMeans {

	public static void main(String[] args) {
		
		// Dataset ou dataframe.
		List<Sample> samples = new ArrayList<>(7);
		samples.add(Sample.of(1, 1));
		samples.add(Sample.of(2, 1));
		samples.add(Sample.of(3, 2));
		samples.add(Sample.of(2, 4.5));
		samples.add(Sample.of(1, 5));
		samples.add(Sample.of(3, 7));
		samples.add(Sample.of(6, 5));
		
		// k = 2
		List<Sample> centroids = List.of(samples.get(0), samples.get(2));
		centroids.get(0).setLabel(1);
		centroids.get(1).setLabel(2);
		
		kmeans(samples, centroids);
		
		System.out.println("Amostras:");
		samples.forEach(System.out::println);
		System.out.println("---------------------");
		
		System.out.println("Centroids:");
		centroids.forEach(System.out::println);
		System.out.println("---------------------");
	}
	
	private static void kmeans(List<Sample> samples, List<Sample> centroids) {
		samples.forEach(sample -> updateSampleLabel(sample, centroids));
		
		List<Sample> centroidsAnteriores = centroids.stream().map(Sample::clone).collect(Collectors.toList());
		updateCentroids(samples, centroids);
		if (hasVariation(centroidsAnteriores, centroids)) {
			kmeans(samples, centroids);
		}
	}

	private static boolean hasVariation(List<Sample> centroidsAnteriores, List<Sample> centroids) {
		double distancia = centroidsAnteriores.get(0).distance(centroids.get(0));
		int i = 1;
		while (distancia == 0 && i < centroids.size()) {
			System.out.println("Distância entre centroids:" + distancia);
			distancia = centroidsAnteriores.get(i).distance(centroids.get(i));
			i++;
		}
		return distancia > 0.0;
	}

	private static void updateCentroids(List<Sample> samples, List<Sample> centroids) {
		// map[1=C1, 2=C2]
		centroids.forEach(Sample::resetData);
		
		Map<Integer, Sample> mapaCentroids = centroids.stream().collect(Collectors.toMap(Sample::getLabel, sample -> sample));
		samples.forEach(sample -> {
			Sample centroid = mapaCentroids.get(sample.getLabel());
			centroid.updateData(sample);
		});
		
		centroids.forEach(Sample::calcMedia);
	}

	private static void updateSampleLabel(Sample sample, List<Sample> centroids) {
		Sample nearestCentroid = centroids.get(0);
		if (centroids.size() > 1) {
			for (int i = 1; i < centroids.size(); i++) {
				if (sample.distance(centroids.get(i)) < sample.distance(nearestCentroid)) {
					nearestCentroid = centroids.get(i);
				}
			}
		}
		sample.setLabel(nearestCentroid.getLabel());		
	}

}
