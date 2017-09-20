/*
 * Copyright (c) 2015 Tata Consultancy Services, Inc. and others.  All rights reserved.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v1.0 which accompanies this distribution,
 * and is available at http://www.eclipse.org/legal/epl-v10.html
 */
package org.opendaylight.kdd.read.demo;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author rashmi.tomer@tcs.com
 */
public class ExampleUsingDL4J_kdd {
	private static Logger log = LoggerFactory
			.getLogger(ExampleUsingDL4J_kdd.class);

	private static final String source = "src/main/resources/classification/";
	private static final String csvFileNameEvalData = "data_eval_kdd.csv";
	private static final String csvFileNameTrainData = "data_train_kdd.csv";

	private static final String csvFilePathTrainData = source
			+ csvFileNameTrainData;
	private static final String csvFilePathEvalData = source
			+ csvFileNameEvalData;

	public static void main(String[] args) throws Exception {
		int seed = 123;
		double learningRate = 0.005;
		int batchSize = 50;
		int nEpochs = 100;

		int numInputs = 38;
		int numOutputs = 4;
		int numHiddenNodes = 20;

		// Load the training data:
		RecordReader rr = new CSVRecordReader();

		rr.initialize(new FileSplit(new File(csvFilePathTrainData)));
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,
				batchSize, 0, 4);

		// Load the test/evaluation data:
		RecordReader rrTest = new CSVRecordReader();

		rrTest.initialize(new FileSplit(new File(csvFilePathEvalData)));
		DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,
				batchSize, 0, 4);

		log.info("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(2)
				.optimizationAlgo(
						OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(learningRate)
				.updater(Updater.NESTEROVS)
				.momentum(0.9)
				.list(2)
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs)
								.nOut(numHiddenNodes)
								.weightInit(WeightInit.NORMALIZED)
								.activation("relu").build())
				.layer(1,
						new OutputLayer.Builder(
								LossFunction.NEGATIVELOGLIKELIHOOD)
								.nIn(numHiddenNodes).nOut(numOutputs)
								.weightInit(WeightInit.NORMALIZED)
								.activation("softmax").build()).pretrain(false)
				.backprop(true).build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		// model.setListeners(new ScoreIterationListener(1));
		model.setListeners(new KddListener(1));

		for (int n = 0; n < nEpochs; n++) {
			model.fit(trainIter);
		}

		System.out.println("Evaluate model....");

		@SuppressWarnings("rawtypes")
		Evaluation eval = new Evaluation(numOutputs);
		int counter = 0;
		while (testIter.hasNext()) {
			DataSet t = testIter.next();
			INDArray features = t.getFeatureMatrix();
			System.out.println("features:" + features.columns());
			// System.out.println("features data:"+features.data());

			INDArray labels = t.getLabels();
			System.out.println("labels:" + labels.columns());
			System.out.println("labels data:" + labels.data());

			INDArray predicted = model.output(features, false);

			System.out.println("predicted:" + predicted.getTrailingOnes());
			System.out.println("predicted data:" + predicted.data());

			eval.eval(labels, predicted);
			counter++;
		}

		// Print the evaluation statistics
		System.out.println(eval.stats());

		System.out.println("Writing confusion matrix to file...");
		try (Writer writer = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(source + "ConfusionMatrix.html"), "utf-8"))) {
			writer.write(eval.getConfusionMatrix().toHTML());
		}
		System.out.println("confusion matrix is availble at file :" + source
				+ "ConfusionMatrix.html");

		// ------------------------------------------------------------------------------------
		// Training is complete. Now plotting the data & predictions only

		// Plot the data
		double xMin = -1.0;
		double xMax = 25;
		double yMin = -1.5;
		double yMax = 5.5;

		// Let's evaluate the predictions at every point in the x/y input
		// space,and plot this in the background
		int nPointsPerAxis = 80;
		double[][] evalPoints = new double[nPointsPerAxis * nPointsPerAxis][38];
		int count = 0;
		for (int i = 0; i < nPointsPerAxis - 1; i++) {
			for (int j = 0; j < nPointsPerAxis; j++) {
				double x = i * (xMax - xMin) / (nPointsPerAxis - 1) + xMin;
				double y = j * (yMax - yMin) / (nPointsPerAxis - 1) + yMin;

				evalPoints[count][0] = x;
				evalPoints[count][1] = y;

				count++;
			}
		}

		INDArray allXYPoints = Nd4j.create(evalPoints);

		INDArray predictionsAtXYPoints = model.output(allXYPoints);

		// Get all of the training data in a single array, and plot it:
		rr.initialize(new FileSplit(new File(csvFilePathTrainData)));
		rr.reset();
		int nTrainPoints = 46;// 2000;
		trainIter = new RecordReaderDataSetIterator(rr, nTrainPoints, 0, 4);
		DataSet ds = trainIter.next();
		PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(),
				allXYPoints, predictionsAtXYPoints, nPointsPerAxis);

		// Get test data, run the test data through the network to generate
		// predictions, and plot those predictions:

		rrTest.initialize(new FileSplit(new File(csvFilePathEvalData)));
		rrTest.reset();
		int nTestPoints = 23;// 1000;
		testIter = new RecordReaderDataSetIterator(rrTest, nTestPoints, 0, 4);
		ds = testIter.next();
		INDArray testPredicted = model.output(ds.getFeatures());
		PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted,
				allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
	}
}
