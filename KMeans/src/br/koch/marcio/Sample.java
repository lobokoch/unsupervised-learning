package br.koch.marcio;

import java.util.Arrays;

public class Sample {
	
	private double[] data;
	private int label;
	private int originalLabel;
	private int size;
	private double silhueta;
	
	public Sample clone() {
		Sample clone = new Sample();
		clone.data = new double[data.length];
		System.arraycopy(data, 0, clone.data, 0, data.length);
		clone.setLabel(label);
		clone.setOriginalLabel(originalLabel);
		clone.setSize(size);
		return clone;
	}
	
	public void resetData() {
		double[] zeros = new double[data.length];
		System.arraycopy(zeros, 0, data, 0, data.length);
		size = 0;
	}
	
	public double distance(Sample sample) {
		// d = sqrt(sum((p1-p2)^2))
		//assert(data.length == sample.length)
		double d = 0.0;
		for (int i = 0; i < data.length; i++) {
			double di = data[i] - sample.data[i];
			d += di * di;
		}
		return Math.sqrt(d);
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}

	public void updateData(Sample sample) {
		for (int i = 0; i < data.length; i++) {
			data[i] += sample.data[i];
		}
		size++;
	}
	
	public void calcMedia() {
		for (int i = 0; i < data.length; i++) {
			data[i] /= size;
		}
	}

	public int getSize() {
		return size;
	}

	public void setSize(int size) {
		this.size = size;
	}

	public static Sample of(String[] data, int labelIndex) {
		Sample sample = new Sample();
		if (labelIndex != -1) {
			sample.setOriginalLabel(Integer.parseInt(data[labelIndex]));
		}
		
		sample.data = new double[labelIndex != -1 ? data.length - 1 : data.length];
		
		int i = 0;
		int j = 0;
		while (i < data.length) {
			if (labelIndex != -1 && i == labelIndex) {
				i++;
				continue;
			}
			sample.data[j] = Double.parseDouble(data[i]);
			i++;
			j++;
		}
		//System.arraycopy(data, srcPos, sample.data, 0, sample.data.length - 1);
		
		return sample;
	}

	public int getOriginalLabel() {
		return originalLabel;
	}

	public void setOriginalLabel(int originalLabel) {
		this.originalLabel = originalLabel;
	}

	public double[] getData() {
		return data;
	}

	public void setData(double[] data) {
		this.data = data;
	}

	public double getSilhueta() {
		return silhueta;
	}

	public void setSilhueta(double silhueta) {
		this.silhueta = silhueta;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("Sample [data=");
		builder.append(Arrays.toString(data));
		builder.append(", label=");
		builder.append(label);
		builder.append(", originalLabel=");
		builder.append(originalLabel);
		builder.append(", size=");
		builder.append(size);
		builder.append(", silhueta=");
		builder.append(silhueta);
		builder.append("]");
		return builder.toString();
	}


}
