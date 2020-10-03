package br.koch.marcio;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

public class KMeans {

	public static void main(String[] args) throws Exception {
		
		// Dataset ou dataframe.
		boolean hasHeader = true;
		String fileName = "D:\\FURB\\Pos_2020\\rec_nao_superv\\DataHotDogs.csv";
		int labelIndex = 0;
		String separator = ",";
		List<Sample> samples = loadData(fileName, hasHeader, labelIndex, separator);
		// kmeans++
		//samples.forEach(System.out::println);
		
		
		int k = 5;
		List<Sample> centroids = getRandomCentroids(k, samples);
		
		centroids.forEach(System.out::println);
		
		kmeans(samples, centroids);
		
		System.out.println("Amostras:");
		samples.forEach(System.out::println);
		System.out.println("---------------------");
		
		System.out.println("Centroids:");
		centroids.forEach(System.out::println);
		System.out.println("---------------------");
	}
	
	private static List<Sample> getRandomCentroids(int k, List<Sample> samples) {
		List<Sample> centroids = new ArrayList<>(k);
		
		Random ran = new Random();
		Set<Integer> usedIndexes = new HashSet<>(k);
		
		while (centroids.size() < k) {
			int index = ran.nextInt(samples.size());
			while (usedIndexes.contains(index)) {
				index = ran.nextInt(samples.size());
			}
			usedIndexes.add(index);
			Sample sample = samples.get(index);
			Sample centroid = sample.clone();
			centroids.add(centroid);
			centroid.setLabel(centroids.size() - 1);
		}
		
		return centroids;
	}

	private static List<Sample> loadData(String fileName, boolean hasHeader, int labelIndex, String separator) throws Exception {
		List<Sample> samples = new ArrayList<>();
		Scanner scanner = new Scanner(new File(fileName));
		// Descartamos o cabeçalho.
		if (hasHeader && scanner.hasNextLine()) {
			scanner.nextLine();
		}
		while (scanner.hasNextLine()) {
			String line = scanner.nextLine();
			if (line != null && !line.isBlank()) {
				String[] data = line.split(separator);
				Sample sample = Sample.of(data, labelIndex);
				samples.add(sample);
			}
		}
		return samples;
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
		System.out.println("Distância entre centroids:" + distancia);
		while (distancia == 0 && i < centroids.size()) {
			distancia = centroidsAnteriores.get(i).distance(centroids.get(i));
			System.out.println("Distância entre centroids:" + distancia);
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
