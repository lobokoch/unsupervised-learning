package br.furb;

import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class PCAEigenFace {
	
	private int numComponents;
	private Mat mean;
	private Mat diffs;
	private Mat covariance;
	private Mat eigenvectors;
	private Mat eigenvalues;
	private Mat eigenFaces;
	private int[] labels;
	private Mat projections;
	
	public PCAEigenFace(int numComponents) {
		this.numComponents = numComponents;
	}
	
	public void train(List<Person> train) {
		calcMean(train);
		calcDiff(train);
		calcCovariance();
		calcEigen();
		calcEigenFaces();
		calcProjections(train);
	}
	
	private void calcProjections(List<Person> train) {
		labels = new int[train.size()];
		projections = new Mat(numComponents, train.size(), CvType.CV_64FC1);
		for (int j = 0; j < diffs.cols(); j++) {
			Mat diff = diffs.col(j);
			Mat w = mul(eigenFaces.t(), diff);
			w.copyTo(projections.col(j));
			labels[j] = train.get(j).getLabel();
		}
		
	}

	private void calcEigenFaces() {
		// Transposição dos autovetores.
		// 1 2 3
		// 4 5 6
		// 1 4
		// 2 5
		// 3 6
		Mat evt = eigenvectors.t();
		Mat ev_k = evt.colRange(0, numComponents > 0 ? numComponents : evt.cols());
		for (int j = 0; j < ev_k.cols(); j++) {
			evt.col(j).copyTo(ev_k.col(j));
		}
		
		eigenFaces = mul(diffs, ev_k);
		for (int j = 0; j < eigenFaces.cols(); j++) {
			Mat ef = eigenFaces.col(j);
			// Normalização L2 = Yi = Xi / sqrt(sum((Xi)^2)), onde i = 0...rows-1
			Core.normalize(ef, ef); 
		}
		
		//printEigenFaces();
		//printEigenValues();
	}

	private void printEigenValues() {
		double sum = 0;
		for (int i = 0; i < eigenvalues.rows(); i++) {
			sum += eigenvalues.get(i, 0)[0];
		}
		
		double acumulado = 0;
		for (int i = 0; i < eigenvalues.rows(); i++) {
			double v = eigenvalues.get(i, 0)[0];
			double percentual = v / sum * 100;
			acumulado += percentual;
			System.out.format("CP%d, percentual:%.2f (%.2f)%n", (i+1), percentual, acumulado);
		}
		
	}

	private void printEigenFaces() {
		for (int j = 0; j < eigenFaces.cols(); j++) {
			Mat y = new Mat(eigenFaces.rows(), 1, eigenFaces.type());
			eigenFaces.col(j).copyTo(y.col(0));
			saveImage(y, "D:\\PCA\\eigenfaces\\eigenface_" + (j+1) + ".jpg"); 
		}
		
	}

	private void calcEigen() {
		eigenvalues = new Mat();
		eigenvectors = new Mat();
		Core.eigen(covariance, eigenvalues, eigenvectors);
	}

	private void calcCovariance() {
		covariance = mul(diffs.t(), diffs);
	}

	private Mat mul(Mat a, Mat b) {
		//A400x6400 * B6400x400
		Mat c = new Mat(a.rows(), b.cols(), CvType.CV_64FC1);
		/*for (int i = 0; i < c.rows(); i++) {
			double v = 0;
			for (int j = 0; j < c.cols(); j++) {				
				for (int k = 0; k < a.cols(); k++) {
					double av = a.get(i, k)[0];
					double bv = b.get(k, j)[0];
					v += av * bv;
				}
				c.put(i, j, v);
			}
		}*/
		
		Core.gemm(a, b, 1, new Mat(), 1, c);
		
		return c;
	}

	private void calcDiff(List<Person> train) {
		Mat sample = train.get(0).getData();
		diffs = new Mat(sample.rows(), train.size(), sample.type() /*CvType.CV_64FC1*/);
		for (int i = 0; i < diffs.rows(); i++) {
			for (int j = 0; j < diffs.cols(); j++) {
				double mv = mean.get(i, 0)[0];
				Mat data = train.get(j).getData();
				double dv = data.get(i, 0)[0];
				double v = dv - mv;
				diffs.put(i, j, v);
			}
		}
		
		/*for (int i = 0; i < train.size(); i++) {
			Core.subtract(train.get(i).getData(), mean, diffs.col(i));
		}*/
	}

	private void calcMean(List<Person> train) {
		Mat sample = train.get(0).getData();
		mean = Mat.zeros(sample.rows(), sample.cols(), sample.type());
		
		train.forEach(person -> {
			Mat data = person.getData();
			for(int i = 0; i < mean.rows(); i++) {
				double mv = mean.get(i, 0)[0];
				double pv = data.get(i, 0)[0];
				mv += pv;
				mean.put(i, 0, mv);
			}
		});
		
		for(int i = 0; i < mean.rows(); i++) {
			double mv = mean.get(i, 0)[0];
			mv /= train.size();
			mean.put(i, 0, mv);
		}
		
		////////////////////////////////////////////////////////////////
		Mat src = new Mat(sample.rows(), train.size(), sample.type());
		for (int i = 0; i < train.size(); i++) {
			train.get(i).getData().col(0).copyTo(src.col(i));
		}
		// 0 1 2
		// 1 2 3
		// 4 5 6
		// 7 8 9
		
		Mat mean2 = Mat.zeros(sample.rows(), 1, sample.type());
		Core.reduce(src, mean2, 1, Core.REDUCE_AVG, mean.type());
		///////////////////////////////////////////////////////////////
		
		//saveImage(mean, "D:\\PCA\\mean.jpg");
		//saveImage(mean2, "D:\\PCA\\mean2.jpg");
		
	}

	private void saveImage(Mat image, String fileName) {
		Mat dst = new Mat();
		Core.normalize(image, dst, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
		
		// 6400 x 1
		// 80 * 80
		dst = dst.reshape(1, 80);
		dst = dst.t();
		Imgcodecs.imwrite(fileName, dst);
	}

	public void predict(Mat testData, int label[], 
			double[] confidence, double[] reconstructionError) {
		
		Mat diff = new Mat();
		Core.subtract(testData, mean, diff);
		
		// Calcula os pesos da imagem desconhecida.
		Mat w = mul(eigenFaces.t(), diff);
		
		// Calcular o vizinho mais próximo dessa projeção "desconhecida".
		int minJ = 0;
		double minDistance = calcDistance(w, projections.col(minJ));
		for (int j = 1; j < projections.cols(); j++) {
			double distance = calcDistance(w, projections.col(j));
			if (distance < minDistance) {
				minDistance = distance;
				minJ = j;
			}
		}
		
		label[0] = labels[minJ];
		confidence[0] = minDistance;
		
		Mat reconstruction = calcReconstruction(w);
		reconstructionError[0] = Core.norm(testData, reconstruction, Core.NORM_L2);
		//saveImage(testData, "D:/PCA/testData.jpg");
		//saveImage(reconstruction, "D:/PCA/reconstruction.jpg");
		//System.out.println("X");
	}

	private Mat calcReconstruction(Mat w) {
		Mat result = mul(eigenFaces, w);
		//result += mean;
		Core.add(result, mean, result);
		
		
		return result;
	}

	private double calcDistance(Mat p, Mat q) {
		// Distância euclidiana.
		// d = sqrt(sum(pi - qi)^2)
		double distance = 0;
		for (int i = 0; i < p.rows(); i++) {
			double pi = p.get(i, 0)[0];
			double qi = q.get(i, 0)[0];
			double d = pi - qi;
			distance += d * d;
		}
		
		double result = Math.sqrt(distance);
		//double result2 = Core.norm(p, q, Core.NORM_L2);
		
		return result;
	}
	

}
