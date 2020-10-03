package br.koch.marcio;

public class Sample {
	
	private double x1;
	private double x2;
	private int label;
	private int size;
	
	public static Sample of(double x1, double x2) {
		Sample sample = new Sample();
		sample.setX1(x1);
		sample.setX2(x2);
		return sample;
	}
	
	public Sample clone() {
		Sample clone = new Sample();
		clone.setX1(x1);
		clone.setX2(x2);
		clone.setLabel(label);
		clone.setSize(size);
		return clone;
	}
	
	public void resetData() {
		x1 = 0.0;
		x2 = 0.0;
		size = 0;
	}
	
	public double getX1() {
		return x1;
	}
	
	public void setX1(double x1) {
		this.x1 = x1;
	}
	
	public double getX2() {
		return x2;
	}
	
	public void setX2(double x2) {
		this.x2 = x2;
	}
	
	

	public double distance(Sample centroide) {
		// d = sqrt(sum((p1-p2)^2))
		double d1 = x1 - centroide.getX1();
		double d2 = x2 - centroide.getX2();
		double d = d1*d1 + d2*d2;
		return Math.sqrt(d);
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("Sample [x1=");
		builder.append(x1);
		builder.append(", x2=");
		builder.append(x2);
		builder.append(", label=");
		builder.append(label);
		builder.append("]");
		return builder.toString();
	}

	public void updateData(Sample sample) {
		x1 += sample.x1;
		x2 += sample.x2;
		size++;
	}
	
	public void calcMedia() {
		x1 /= size;
		x2 /= size;
	}

	public int getSize() {
		return size;
	}

	public void setSize(int size) {
		this.size = size;
	}
	

}
