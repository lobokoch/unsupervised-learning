package br.com.furb;

import java.awt.Image;
import java.util.Timer;
import java.util.TimerTask;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class FaceDetectionWithCamera {
	
	private CascadeClassifier faceClassifier;
	private CascadeClassifier eyeClassifier;
	private CascadeClassifier smileClassifier;
	
	private VideoCapture camera;
	
	private JFrame frame;
	private JPanel panel;
	private Image image;
	
	private Timer timer;
	
	public FaceDetectionWithCamera() {
		init();
	}
	
	public void start() {
		long delay = 0;
		long period = 33;
		
		timer.schedule(new TimerTask() {
			
			@Override
			public void run() {
				Mat img = new Mat();
				camera.read(img);
				detectAndShow(img);
			}
			
		}, delay, period);
	}
	
	private void detectAndShow(Mat img) {
		
		MatOfRect faceObjects = new MatOfRect();
		faceClassifier.detectMultiScale(img, faceObjects);
		
		MatOfRect eyesObjects = new MatOfRect();
		eyeClassifier.detectMultiScale(img, eyesObjects);
		
		MatOfRect smileObjects = new MatOfRect();
		smileClassifier.detectMultiScale(img, smileObjects, 1.1, 10, 0, new Size(), new Size());
		
		faceObjects.toList().forEach(rect -> {
			Point p1 = new Point(rect.x, rect.y);
			Point p2 = new Point(rect.x + rect.width, rect.y + rect.height);
			Rect rec = new Rect(p1, p2);
			Scalar color = new Scalar(0, 0, 255);
			Imgproc.rectangle(img, rec, color, 3);
		});
		
		smileObjects.toList().forEach(rect -> {
			Point p1 = new Point(rect.x, rect.y);
			Point p2 = new Point(rect.x + rect.width, rect.y + rect.height);
			Rect rec = new Rect(p1, p2);
			Scalar color = new Scalar(0, 255, 0);
			Imgproc.rectangle(img, rec, color, 3);
		});
		
		eyesObjects.toList().forEach(rect -> {
			Point p1 = new Point(rect.x, rect.y);
			Point p2 = new Point(rect.x + rect.width, rect.y + rect.height);
			Rect rec = new Rect(p1, p2);
			Scalar color = new Scalar(255, 255, 255);
			Imgproc.rectangle(img, rec, color, 3);
		});
		
		this.image = HighGui.toBufferedImage(img);
		panel.repaint();
	}

	private void init() {
		final String cascadePath = "D:\\mk\\OpenCV440\\install\\etc\\haarcascades\\";
		
		faceClassifier = new CascadeClassifier(cascadePath + "haarcascade_frontalface_alt.xml");
		eyeClassifier = new CascadeClassifier(cascadePath + "haarcascade_eye.xml");
		smileClassifier = new CascadeClassifier(cascadePath + "haarcascade_smile.xml");
		
		camera = new VideoCapture();
		camera.open(0);
		
		buildFrame();
		
		timer = new Timer();
		
	}

	private JFrame buildFrame() {
		frame = new JFrame();
		
		frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		frame.setSize(800, 600);
		frame.setTitle("Face Detection with OpenCV.");
		frame.setLocationRelativeTo(null);
		
		panel = new JPanel() {
			private static final long serialVersionUID = 1L;
			
			@Override
			protected void paintComponent(java.awt.Graphics g) {
				super.paintComponent(g);
				
				if (image != null) {
					g.drawImage(image, 0, 0, 800, 600, null);
				}
				
			}
		};//
		
		frame.add(panel);
		
		frame.setVisible(true);
		
		return frame;		
	}

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		FaceDetectionWithCamera faceDetection = new FaceDetectionWithCamera();
		faceDetection.start();
	}

}
