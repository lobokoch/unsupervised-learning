package br.furb;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class PCA {

	public static void main(String[] args) {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		String path = "D:\\PCA\\dataset\\ORL\\";
		List<Person> train = new ArrayList<>();
		List<Person> test = new ArrayList<>();
		int p = 7; // holdout de 70/30, treino/teste
		loadDataset(path, train, test, p);
		
		//test.add(toPerson("D:\\dataset\\orl\\1002_182.jpg"));
		//test.add(toPerson("D:\\dataset\\orl\\1003_182.jpg"));
		//test.add(toPerson("D:\\dataset\\orl\\7000_777.jpg"));
		//test.add(toPerson("D:\\dataset\\orl\\9000_999.jpg"));
		//test.add(toPerson("D:\\dataset\\orl\\9001_999.jpg"));
		//test.add(toPerson("D:\\dataset\\orl\\8000_888.jpg"));
		
		final int startComps = 15;
		final int maxComps = 15;
		
		double minDistance = Double.MAX_VALUE;
		double maxDistance = Double.MIN_VALUE;
		double meanDistance = 0;
		int corrects = 0;
		
		///// Using pure OpenCV ///////////////////////////////////////////////////
		double minDistance2 = Double.MAX_VALUE;
		double maxDistance2 = Double.MIN_VALUE;
		double meanDistance2 = 0;
		int corrects2 = 0;
		///////////////////////////////////////////////////////////////////////////
		
		double minRec = Double.MAX_VALUE;
		double maxRec = Double.MIN_VALUE;
		double meanRec = 0;
		
		final double MAX_DISTANCE = 2500;
		final double MAX_REC = 2900;
		
		for (int numComponents = startComps; numComponents <= maxComps; numComponents++) {
			PCAEigenFace model = new PCAEigenFace(numComponents);
			model.train(train);
			
			///// Using pure OpenCV ///////////////////////////////////////////////////
			EigenFaceRecognizer model2 = EigenFaceRecognizer.create(numComponents);
			List<Mat> src = new ArrayList<>(train.size());
			Mat labels = new Mat(train.size(), 1, CvType.CV_32SC1);
			for (int i = 0; i < train.size(); i++) {
				Person person = train.get(i);
				src.add(person.getData());
				labels.put(i, 0, person.getLabel());
			}
			model2.train(src, labels);
			///////////////////////////////////////////////////////////////////////////
			
			int truePositiveCount = 0;
			int trueNegativesCount = 0;
			for (Person personToTest: test) {
				Mat testData = personToTest.getData();
				int[] label = new int[1]; 
				double[] confidence = new double[1]; 
				double[] reconstructionError = new double[1];
				
				model.predict(testData, label, confidence, reconstructionError);
				
				///// Using pure OpenCV ///////////////////////////////////////////////////
				
				int[] label2 = new int[1];
				double[] confidence2 = new double[1];
				model2.predict(testData, label2, confidence2);
				if (personToTest.getLabel() == label2[0]) {
					corrects2++;
				}
				if (confidence2[0] < minDistance2) {
					minDistance2 = confidence2[0]; 
				}
				if (confidence2[0] > maxDistance2) {
					maxDistance2 = confidence2[0]; 
				}
				meanDistance2 += confidence2[0];
				///////////////////////////////////////////////////////////////////////////
				
				boolean labelOK = label[0] == personToTest.getLabel(); 
				if (labelOK) {
					corrects++;
				}
				
				if (reconstructionError[0] > MAX_REC) {
					
					System.out.println("NOT A PERSON - Predicted label:" + label[0] + 
							", confidence:" + confidence[0] + 
							", reconstructedError:" + reconstructionError[0] +
							", original label: " + personToTest.getLabel());
					
					if (!labelOK) {
						trueNegativesCount++;
					}
				} else if (confidence[0] > MAX_DISTANCE) {
					
					System.out.println("UKNOWN PERSON (by distance) - Predicted label:" + label[0] + 
							", confidence:" + confidence[0] + 
							", reconstructedError:" + reconstructionError[0] +
							", original label: " + personToTest.getLabel());
					
					if (!labelOK) {
						trueNegativesCount++;
					}
					
				} else if (reconstructionError[0] > 2400 && confidence[0] > 1800) {
					System.out.println("UKNOWN PERSON (by two factors) - Predicted label:" + label[0] + 
							", confidence:" + confidence[0] + 
							", reconstructedError:" + reconstructionError[0] +
							", original label: " + personToTest.getLabel());
					
					if (!labelOK) {
						trueNegativesCount++;
					}
				}
				else if (labelOK) {
					truePositiveCount++;
				}
				else {
					System.out.println("UNKNOWN - Predicted label:" + label[0] + 
						", confidence:" + confidence[0] + 
						", reconstructedError:" + reconstructionError[0] +
						", original label: " + personToTest.getLabel());
				}
				
				if (personToTest.getLabel() <= 40) {
					// definir um limiar de confiança/distância de confiança
					if (confidence[0] < minDistance) {
						minDistance = confidence[0];
					}
					
					if (confidence[0] > maxDistance) {
						maxDistance = confidence[0];
					}
					
					meanDistance += confidence[0];
					
					// definir um limiar de confiança/distância de confiança
					if (reconstructionError[0] < minRec) {
						minRec = reconstructionError[0];
					}
					
					if (reconstructionError[0] > maxRec) {
						maxRec = reconstructionError[0];
					}
					
					meanRec += reconstructionError[0];
				}
				
				
				
			} //for
			
			int trues = truePositiveCount + trueNegativesCount;
			
			double accuracy = (double)trues / test.size() * 100;
			
			
			
			System.out.format("numComponents:%d, Percentual de acerto:%.2f (%d de %d)%n", 
					numComponents, accuracy, truePositiveCount, test.size());
			
			System.out.format("truePositiveCount:%d, trueNegativesCount:%d%n", truePositiveCount, trueNegativesCount);
			
			System.out.format("minDistance:%.2f, maxDistance:%.2f, meanDistance: %.2f%n", minDistance, maxDistance, meanDistance / test.size());
			System.out.format("minRec:%.2f, maxRec:%.2f, meanRec: %.2f%n", minRec, maxRec, meanRec / test.size());
			
			System.out.println("corrects: " + corrects);
			
			System.out.println("**********  BY OpenCV **************");
			System.out.println("corrects2: " + corrects2);
			System.out.format("minDistance2:%.2f, maxDistance2:%.2f, meanDistance2: %.2f%n", 
					minDistance2, maxDistance2, meanDistance2 / test.size());
		}
		
	}

	private static void loadDataset(String path, List<Person> train, List<Person> test, int p) {
		File folder = new File(path);
		File[] filesArray = folder.listFiles((dir, fileName) -> fileName.toLowerCase().endsWith(".jpg"));
		List<Person> people = Arrays.asList(filesArray)
				.stream()
				.map(file -> toPerson(file.getPath())).collect(Collectors.toList());
		
		people.sort(Comparator.comparing(Person::getId));
		// 
		Random ran = new Random();
		final int numSamplesPerPerson = 10;
		List<Person> samples = new ArrayList<>(numSamplesPerPerson);
		people.forEach(person -> {
			samples.add(person);
			if (samples.size() == numSamplesPerPerson) {
				while (samples.size() > p) {
					int index = ran.nextInt(samples.size());
					test.add(samples.remove(index));
				}
				
				if (p == numSamplesPerPerson) {
					test.addAll(samples);
				}
				
				train.addAll(samples);
				samples.clear();
			}
		});
			
	}

	private static Person toPerson(String fileName) {
		Person person = new Person();
		
		//D:\xx\1_1.jpg
		String dataPart = fileName.substring(fileName.lastIndexOf("\\") + 1, fileName.lastIndexOf(".jpg"));
		String[] data = dataPart.split("_");
		
		person.setId(Integer.parseInt(data[0]));
		person.setLabel(Integer.parseInt(data[1]));
		
		person.setData(getImageData(fileName));
		
		return person;
	}

	private static Mat getImageData(String fileName) {
		Mat img = Imgcodecs.imread(fileName, Imgcodecs.IMREAD_GRAYSCALE);
		
		// Muda o tamanho para 80x80
		Mat dst = new Mat();
		Imgproc.resize(img, dst, new Size(80, 80));
		
		//Converter para vetor coluna
		// 1 2
		// 3 4
		//1 3
		//2 4
		// 1
		// 3
		// 2
		// 4
		dst = dst.t().reshape(1, dst.cols() * dst.rows());
		
		// Converte de 8 bits sem sinal, para 64 bits com sinal, preserva 1 canal apenas.
		Mat data = new Mat();
		dst.convertTo(data, CvType.CV_64FC1);
		
		return data;
	}
	


}
