package br.koch.marcio;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
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
		
		
		double[] coefSilhueta = new double[1];
		
		//centroids.forEach(System.out::println);
		
		int kInicial = 2;
		int kFinal = 7;
		Map<Integer, Double> coeficientes = new HashMap<>();
		for (int k = kInicial; k <= kFinal; k++) {
			List<Sample> centroids = getRandomCentroids(k, samples);
			System.out.println("Calculnado k-means de k:" + k);
			kmeans(samples, centroids);
			silhueta(samples, centroids, k, coeficientes);
		}
		
		double[] maxMean = new double[1];
		maxMean[0] = Double.MIN_VALUE;
		int[] bestK = new int[1];
		bestK[0] = -1;
		coeficientes.forEach((k, soma) -> {
			double mean = soma / samples.size();
			//coeficientes.put(key, mean);
			if (mean > maxMean[0]) {
				maxMean[0] = mean;
				bestK[0] = k;
			}
		});
		
		System.out.println("Best k is:" + bestK[0]);
		
		/*System.out.println("Amostras:");
		samples.forEach(System.out::println);
		System.out.println("---------------------");
		
		System.out.println("Centroids:");
		centroids.forEach(System.out::println);
		System.out.println("---------------------");*/
	}
	
	private static void silhueta(List<Sample> samples, List<Sample> centroids, int k, Map<Integer, Double> coeficientes) {
		// Calcular a média do ponto ai com os seus pontos colegas de cluster.
		samples.forEach(sample -> {
			Map<Integer, Sample> mapaCentroids = centroids.stream()
					.collect(Collectors.toMap(Sample::getLabel, c -> c));
			
			double a = calcMedia(sample, samples, mapaCentroids.remove(sample.getLabel()));
			double b = Double.MAX_VALUE;
			Iterator<Integer> iterator = mapaCentroids.keySet().iterator();
			while (iterator.hasNext()) {
				double x = calcMedia(sample, samples, mapaCentroids.get(iterator.next()));
				if (x < b) {
					b = x;
				}
			}
			
			double s = (b - a) / Math.max(a, b);
			sample.setSilhueta(s);
			
			coeficientes.put(k, coeficientes.getOrDefault(k, 0.0) + s);
		});
		
		/*coeficientes.forEach((key, value) -> {
			double mean = value / samples.size();
			coeficientes.put(key, mean);			
		});*/
		
		centroids.forEach(c -> {
			
			List<Sample> pts = samples.stream().filter(it -> it.getLabel() == c.getLabel())
					.collect(Collectors.toList());
			
			pts.forEach(pt -> {
				int si = (int) Math.ceil(pt.getSilhueta() * 100);
				if (si >= 0) {
					for (int i = 0; i < 100; i++) {
						System.out.print(i == 100 - 1 ? "|" : " ");
					}
				} else {
					si = si * -1;
					int max = 100 - si;
					for (int i = 0; i < max - 1; i++) {
						System.out.print(" ");
					}
				}
				for (int i = 0; i < si; i++) {
					System.out.print(c.getLabel());
				}
				if (pt.getSilhueta() < 0) {
					System.out.print("|");
				}
				System.out.print(" " + String.format("%.2f", pt.getSilhueta()));
				System.out.println(" ");
			});
			System.out.println(" ");
		});
		
		
	}

	private static double calcMedia(Sample sample, List<Sample> samples, Sample centroid) {
		// Coleta todas as amostras do Ci
		List<Sample> samplesDoCluster = samples.stream()
				.filter(it -> it.getLabel() == centroid.getLabel())
				.collect(Collectors.toList());
		
		// Remove a amostra i do conjunto de amostars do cluster Ci.
		samplesDoCluster.remove(sample);
		
		double[] sum = new double[1];
		samplesDoCluster.forEach(it -> {
			sum[0] += sample.distance(it);
		});
		
		return sum[0] /= samplesDoCluster.size();
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
		//System.out.println("Distância entre centroids:" + distancia);
		while (distancia == 0 && i < centroids.size()) {
			distancia = centroidsAnteriores.get(i).distance(centroids.get(i));
			//System.out.println("Distância entre centroids:" + distancia);
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
