/*
 *    EvaluateNonStationaryDynamicStream.java
 *    
 *    Copyright (C) 2013 University of Texas at Dallas
 *    @author Brandon S. Parker (brandon.parker@utdallas.edu)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.tasks;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.net.URISyntaxException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Random;
import java.util.Set;
import moa.classifiers.Classifier;
import moa.classifiers.novelClass.AbstractNovelClassClassifier;
import moa.core.Measurement;
import moa.core.MultiClassConfusionMatrix;
import moa.core.ObjectRepository;
import moa.core.TimingUtils;
import moa.evaluation.ClassificationPerformanceEvaluator;
import moa.evaluation.LearningCurve;
import moa.evaluation.LearningEvaluation;
import moa.options.ClassOption;
import moa.options.FileOption;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.streams.InstanceStream;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Evaluates the classification algorithms with provisions for outlier, novel class, and semi-supervised approaches
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 8 $
 */
public class EvaluateNonStationaryDynamicStream extends MainTask {


    protected static final DateFormat iso8601FormatString = new SimpleDateFormat("yyyyMMdd'T'HHmmss");

    private static final long serialVersionUID = 1L;

    /**
     * Allows to select the trained classifier.
     */
    public ClassOption learnerOption = new ClassOption("learner", 'L',
                                                       "Classifier to train.",
                                                       moa.classifiers.Classifier.class,
                                                       "moa.classifiers.meta.M3");
                                                       //"moa.classifiers.bayes.NaiveBayes");

    /**
     * Allows to select the stream the classifier will learn.
     */
    public ClassOption streamOption = new ClassOption("stream", 's',
                                                      "Stream to learn from.",
                                                      moa.streams.InstanceStream.class,
                                                      "moa.streams.generators.InducedRandomNonStationaryDataGenerator");

    /**
     * Allows to select the classifier performance evaluation method.
     */
    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
                                                         "Classification performance evaluation method.",
                                                         moa.evaluation.ClassificationPerformanceEvaluator.class,
                                                         "moa.evaluation.ClassificationWithNovelClassPerformanceEvaluator");

    public FlagOption sendZeroWeightsOption = new FlagOption("sendZeroWeights",'z',
                                                        "Send non-training data as zero-weight instances to allow SSL");
    
    /**
     * Fraction of data to use for training.
     */
    public FloatOption trainingFractionOption = new FloatOption("trainingFraction", 'p',
                                                                "Fraction of data to use for training (semi-supervised)",
                                                                1.0, 0.00, 1.00);
    
    /**
     * Allow to define the training/testing chunk size.
     */
    public IntOption chunkSizeOption = new IntOption("chunkSize", 'c',
                                                     "Number of instances in a data chunk. Note a chunk size of 1 indicates immediate 'test then train' eval cycle",
                                                     1, 1, Integer.MAX_VALUE);

    /**
     * Defines how many instances to 'skip' evaluations for as a warm-up phase
     */
    public IntOption warmupSampleSizeOption = new IntOption("warmupSize", 'w',
                                                            "How many pure trainig instances to ignore up front before capturing statistics.",
                                                            7000, 0, Integer.MAX_VALUE);

    /**
     * Establishes delay between ingest of a data instance and when that instance is used for training
     *
     */
    public IntOption trainingTimeDelayOption = new IntOption("trainingDelay", 'l',
                                                             "Time Delay (latency) for using a labeled instance for training purposes",
                                                             2000, 0, Integer.MAX_VALUE);

    /**
     * Establishes delay between ingest of a data instance and when that instance is used for training
     *
     */
    public IntOption labelDeadlineOption = new IntOption("labelingDeadline", 'T',
                                                         "Maximum 'time' (number instances seen) to delay until outlier labels are finalized",
                                                         100, 1, Integer.MAX_VALUE);

    /**
     * Defines how often classifier parameters will be calculated.
     */
    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency", 'f',
                                                           "How many instances between samples of the learning performance.",
                                                           5000, 0, Integer.MAX_VALUE);
    
    
    /**
     * Random Seed for RNG.
     */
    public IntOption randomSeed = new IntOption("randomSeed", 'r',
                                                "Random Seed for RNG.",
                                                42, 0, Integer.MAX_VALUE);

    /**
     * Allows to define the output file name and location.
     */
    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
                                                      "File to append intermediate csv results to.",
                                                      "ENSDS-" + iso8601FormatString.format(new Date()) + "-metrics.csv", "csv", true);

    /**
     * Allows to define the output file name and location.
     */
    public FileOption confusionMatrixFileOption = new FileOption("confusionMatrixFile", 'm',
                                                      "File to write confusionm atrix to",
                                                      "", "csv", true);
    
    /**
     * Allows to define the maximum number of seconds to test/train for (-1 = no limit).
     */
    public IntOption timeLimitOption = new IntOption("timeLimit", 't',
                                                     "Maximum number of seconds to test/train for (-1 = no limit).", -1,
                                                     -1, Integer.MAX_VALUE);
    /**
     * Allows to define the maximum number of instances to test/train on (-1 = no limit).
     */
    public IntOption instanceLimitOption = new IntOption("instanceLimit", 'i',
                                                         "Maximum number of instances to test/train on  (-1 = no limit).",
                                                         5000000, -1, Integer.MAX_VALUE);
    
        /**
     * Allows to define the memory limit for the created model.
     */
    public IntOption maxMemoryOption = new IntOption("maxMemory", 'b',
                                                     "Maximum size of model (in bytes). -1 = no limit.",
                                                     -1, -1, Integer.MAX_VALUE);

    /**
     * Allows to define the frequency of memory checks.
     */
    public IntOption memCheckFrequencyOption = new IntOption("memCheckFrequency", 'q',
                                                             "How many instances between memory bound checks.",
                                                             100000, 0, Integer.MAX_VALUE);
   
    
    
    /**
     * The main classification algorithm under test
     */
    private Classifier learner = null;

    /**
     * The data stream used in the test harness
     */
    private InstanceStream stream = null;

    /**
     * The test evaluation method
     */
    private ClassificationPerformanceEvaluator evaluator = null;

    /**
     * High level metrics for number of labels seen
     */
    protected int knownLabels[] = null;
    
    protected class TimeBoxedInstance {
        public Instance inst = null;
        public long startTime = 0;
        public long deadline = 0;
        public double[] priorVotes = null;
        public TimeBoxedInstance(Instance x, long s, long d, double[] h) {
            inst = x;
            startTime = s;
            deadline = s + d;
            priorVotes = h;
        }
    }
        
    /**
     * Instance cache for latent training purposes
     */
    private final LinkedList<TimeBoxedInstance> latentTrainingInstQueue = new LinkedList<>();

    /**
     * Instance cache for latent Novel Class declaration of outliers
     */
    private final LinkedList<TimeBoxedInstance> pendingFinalLabelInstQueue = new LinkedList<>();

    /**
     * Stream for results as they are found
     */
    private PrintStream immediateResultStream = null;

    /**
     * Tally of the number of data instances processed
     */
    private long instancesProcessed = 0;

    /**
     * Tracking of time that has passed (CPU Time)
     */
    private int secondsElapsed = 0;

    /**
     * Output file to dump the metrics from this test
     */
    private File dumpFile = null;

    /**
     * Random Number Generator (RNG)
     */
    private Random rng = null;

    /**
     * Task Monitor to report back task status
     */
    protected TaskMonitor monitor;

    // Constants for "magic numbers"
    public static final double BYTES_TO_GIGABYTES = 1.0 / (1024.0 * 1024.0 * 1024.0);
    public static final double SECONDS_TO_HOURS = 1.0 / 3600.0;
    public static final int EXTRA_LABELS_TO_ADD = 2;
    public static final double NOVEL_WEIGHT = 0.0;
    
    protected boolean firstDump = true;
    protected boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
    protected boolean inWarmupPhase = true;
    protected double RAMHours = 0.0;
    protected long samplesTested = 0, samplesTrained = 1;
    protected long sampleTestTime = 0, sampleTrainTime = 1;
    protected long startTime = 0;

    protected MultiClassConfusionMatrix cm = new MultiClassConfusionMatrix();

    
    /**
     * Defines the task's result type.
     *
     * @return task type
     */
    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }

    /**
     * Prints basic information to the logger
     */
    private void flowerbox() {
        String compileDateStr = "(unknown)";
        try {
            File jarFile = new File(this.getClass().getProtectionDomain().getCodeSource().getLocation().toURI());
            Date compileDate = new Date(jarFile.lastModified());
            compileDateStr = compileDate.toString();
        } catch (URISyntaxException e) { }

        System.out.println(("+------------------------------------+"));
        System.out.println(("| EvaluateNonStationaryDynamicStream |"));
        System.out.println(("|    Version 8                       |"));
        System.out.println(("+------------------------------------+"));
        System.out.println(("Build Date: " + compileDateStr));
        System.out.println(("OS:         " + System.getProperty("os.name") + " (" + System.getProperty("os.arch") + " Vers." + System.getProperty("os.version") + ")"));
        System.out.println(("JVM:        " + System.getProperty("java.vm.vendor") + " " + System.getProperty("java.vm.name") + " " + System.getProperty("java.vm.version")));
        System.out.println(("JRE:        " + System.getProperty("java.vendor") + " " + System.getProperty("java.version")));
        System.out.println(("User:       " + System.getProperty("user.name")));
        System.out.println(("Num Proc:   " + Runtime.getRuntime().availableProcessors()));
        System.out.println(("Initial Mem:" + Runtime.getRuntime().totalMemory()));
        System.out.println(("Max Memory: " + Runtime.getRuntime().maxMemory()));
        System.out.println(("Current Working Directory: " + System.getProperty("user.dir")));
        System.out.println(("Algorithm: " + this.learnerOption.getValueAsCLIString()));
        //System.out.println(("Description:" + learner.getPurposeString()));
        System.out.println(("Data File: " + this.streamOption.getValueAsCLIString()));
    }

    @Override
    protected Object doMainTask(TaskMonitor p_monitor, ObjectRepository repository) {
        
        flowerbox();
        reset();
        this.monitor = p_monitor;
        this.monitor.setCurrentActivity("Evaluating learner...", -1.0);
        this.monitor.setCurrentActivityDescription("Evaluating " + this.learner.getClass().getSimpleName());
        this.monitor.setCurrentActivityFractionComplete(0);
        //this.monitor.setLatestResultPreview("Preview This");
        
        this.rng = new Random(this.randomSeed.getValue());
        this.latentTrainingInstQueue.clear();
        this.pendingFinalLabelInstQueue.clear();
        LearningCurve learningCurve = new LearningCurve("learning evaluation instances");
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        this.cm = new MultiClassConfusionMatrix((this.learnerOption.getValueAsCLIString() + " on " + this.streamOption.getValueAsCLIString()));
        // Run through entire data set...
        while (stream.hasMoreInstances()
                && ((this.instanceLimitOption.getValue() < 0) || (instancesProcessed < this.instanceLimitOption.getValue()))
                && ((this.timeLimitOption.getValue() < 0) || (secondsElapsed < this.timeLimitOption.getValue()))) {
            this.inWarmupPhase = (this.instancesProcessed < this.warmupSampleSizeOption.getValue());
            
            this.processChunk(getChunk(), learningCurve, evaluateStartTime);
            if (!memoryTesting(monitor, learningCurve, evaluateStartTime)) {
                return null;
            }
        } //end while()
        
        // Wrap up...
        if (this.confusionMatrixFileOption.getValue().length() > 0) {
            this.monitor.setCurrentActivityDescription("Writing Confusion Matrix");
            this.cm.writeCSV(this.confusionMatrixFileOption.getValue());
        }
        this.monitor.setCurrentActivityDescription("Done.");
        this.monitor.requestResultPreview();
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        this.monitor.setCurrentActivityDescription("Done.");
        return learningCurve;
    } //end doMainTask()

    /**
     * Manage testing, training, and the reporting thereof
     *
     * @param D Data set (or "chunk") to process
     * @param learningCurve for graphing
     * @param evaluateStartTime for tracking processing time
     */
    protected void processChunk(Instances D, LearningCurve learningCurve, long evaluateStartTime) {
        // Certain algorithms (like AHOT) have issue when the training data set is too small and they emit an exception
        // So we need to wrap a TRY/Catch pattern around the processing to gracefully handle those issues
        if (knownLabels == null) {
            knownLabels = new int[D.firstInstance().numClasses() + 2];
            Arrays.fill(knownLabels, 0);
        }
        //try {
              
            // TEST all data instances in stream...
            if (inWarmupPhase) {
                try {
                    for(Instance x : D) {
                        this.evaluator.addResult(x, null);
                        this.knownLabels[(int) x.classValue()]++;
                    }
                } catch (Exception e) {}; // don't care, just avoid problems with sending null votes if not our own evaluator
            } else {
                startTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                samplesTested += test(D);
                sampleTestTime += TimingUtils.getNanoCPUTimeOfCurrentThread() - startTime;
            }
            
            // Train models adhering to latency and semi-supervised reduced training parameters...
            startTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
            samplesTrained += train();
            sampleTrainTime += TimingUtils.getNanoCPUTimeOfCurrentThread() - startTime;
       // } catch (Exception e) {
       //     System.err.println("Caught Exception: " + e.toString() + " (" + e.getCause() + ": " +  e.getMessage()+ ")");
       // }

        // Result output and MOA framework housekeeping...
        if (!inWarmupPhase && (this.instancesProcessed % this.sampleFrequencyOption.getValue() == 0)) {
            double RAMHoursIncrement = learner.measureByteSize() * BYTES_TO_GIGABYTES; //GBs
            RAMHoursIncrement *= (TimingUtils.nanoTimeToSeconds(sampleTrainTime + sampleTestTime) / SECONDS_TO_HOURS); //Hours
            RAMHours += RAMHoursIncrement;

            double avgTrainTime = TimingUtils.nanoTimeToSeconds(sampleTrainTime) / ((double) this.sampleFrequencyOption.getValue() / samplesTrained);
            double avgTestTime = TimingUtils.nanoTimeToSeconds(sampleTestTime) / ((double) this.sampleFrequencyOption.getValue() / samplesTested);
            learningCurve.insertEntry(new LearningEvaluation(
                    new Measurement[]{
                        new Measurement("learning evaluation instances", instancesProcessed),
                        new Measurement("evaluation time (" + (preciseCPUTiming ? "cpu " : "") + "seconds)", TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime)),
                        new Measurement("model cost (RAM-Hours)", RAMHours),
                        new Measurement("average chunk train time", avgTrainTime),
                        new Measurement("average chunk train speed", samplesTrained / avgTrainTime),
                        new Measurement("average chunk test time", avgTestTime),
                        new Measurement("average chunk test speed", samplesTested / avgTestTime)
                        
                        },
                    this.evaluator,
                    this.learner));

            if (immediateResultStream != null) {
                if (firstDump) {
                    immediateResultStream.println(learningCurve.headerToString());
                    firstDump = false;
                }
                immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                immediateResultStream.flush();
            }
            samplesTested = 0;
            sampleTestTime = 0;
            samplesTrained = 0;
            sampleTrainTime = 0;
        }
    }

    /**
     *
     *
     * @return instances retrieved from stream
     */
    private Instances getChunk() {
        Instances chunk = new Instances(stream.getHeader(), this.chunkSizeOption.getValue());
        // Add "chunk size" number of instances to test directly from the stream (first time we see each instance):
        while (stream.hasMoreInstances() && chunk.numInstances() < this.chunkSizeOption.getValue()) {
            Instance inst = stream.nextInstance();
            this.instancesProcessed++;
            chunk.add(inst);
            
            if (this.inWarmupPhase) { // For warmup phase, use full and immediate training
                inst.setWeight(1.0);
                latentTrainingInstQueue.addFirst(new TimeBoxedInstance(inst,this.instancesProcessed, 0,null)); 
            } else if (rng.nextFloat() > this.trainingFractionOption.getValue()) { // Select a portion for latent training set by setting non-training instance weight to zero.
                // place at beginning of the queue/list and record intended activation 'time' for immediate unsupervised 'training'
                inst.setWeight(0.0);
                latentTrainingInstQueue.addFirst(new TimeBoxedInstance(inst,this.instancesProcessed, 0,null)); 
            } else {
                if (this.sendZeroWeightsOption.isSet()) {
                    Instance unsupervisedInstance = (Instance) inst.copy();
                    unsupervisedInstance.setWeight(0.0);
                    //unsupervisedInstance.setClassValue(0);
                    latentTrainingInstQueue.addFirst(new TimeBoxedInstance(unsupervisedInstance, this.instancesProcessed, 0, null));
                }
                // place at end of the queue/list and record intended activation 'time' for latent supervised training
                latentTrainingInstQueue.addLast(new TimeBoxedInstance(inst, this.instancesProcessed, this.trainingTimeDelayOption.getValue(), null));
            }

            // MOA framework housekeeping and reporting...
            if ((instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES) == 0) {
                this.monitor.setCurrentActivityDescription("Updating Metrics");
                if (monitor.taskShouldAbort()) {
                    chunk.clear();
                    return chunk;
                }
                long estimatedRemainingInstances = stream.estimatedRemainingInstances();

                if (this.instanceLimitOption.getValue() > 0) {
                    long maxRemaining = this.instanceLimitOption.getValue() - instancesProcessed;
                    if ((estimatedRemainingInstances < 0) || (maxRemaining < estimatedRemainingInstances)) {
                        estimatedRemainingInstances = maxRemaining;
                    }
                }
                monitor.setCurrentActivityFractionComplete((double) instancesProcessed / (double) (instancesProcessed + estimatedRemainingInstances));
            }
        } // end while
        return chunk;
    }

    /**
     *
     * @param testInstances instance set to evaluate accuracy
     * @return number of instances actually tested
     */
    private int test(Instances testInstances) {
        this.monitor.setCurrentActivityDescription("Testing Instances");
        int ret = testInstances.size();
        int novelClassLabel = testInstances.numClasses();
        int outlierLabel = novelClassLabel + 1;
        
        // For latent label outliers that have reached their deadline, we must now make a decision:
        while (!this.pendingFinalLabelInstQueue.isEmpty() && this.pendingFinalLabelInstQueue.peek().deadline <= this.instancesProcessed) {
            TimeBoxedInstance ti = this.pendingFinalLabelInstQueue.pop();
            int y = (int) ti.inst.classValue();
            double[] prediction = null;
            if (y >= 0 && y < knownLabels.length && knownLabels[y] <= this.labelDeadlineOption.getValue()) {
                Instance novelInst = (Instance) ti.inst.copy();
                //novelInst.setDataset(AbstractNovelClassClassifier.augmentInstances(novelInst.dataset()));
                //novelInst.setClassValue(AbstractNovelClassClassifier.NOVEL_LABEL_STR);
                novelInst.setWeight(NOVEL_WEIGHT);
                prediction = learner.getVotesForInstance(novelInst);
                evaluator.addResult(novelInst, prediction); // Outlier out of time. Remove it
            } else {
                prediction = learner.getVotesForInstance(ti.inst);
                evaluator.addResult(ti.inst, prediction); // Outlier out of time. Remove it
            }
            
            this.cm.add(weka.core.Utils.maxIndex(prediction),ti.inst.classValue());
        }
               
        // Run accuracy test for current instance(s)
        for (Instance i : testInstances) {
            int y = (int) i.classValue();
            double[] prediction = null;
            Instance instToActuallyPredict = i;
            // If novel, make a special instance
            if (y >= 0 && y < knownLabels.length && knownLabels[y] <= this.labelDeadlineOption.getValue()) {
                instToActuallyPredict = (Instance) i.copy();
                //novelInst.setDataset(AbstractNovelClassClassifier.augmentInstances(novelInst.dataset()));
                //novelInst.setClassValue(AbstractNovelClassClassifier.NOVEL_LABEL_STR); // WARNING - this crashes other algorithms if not also done on training!
                instToActuallyPredict.setWeight(NOVEL_WEIGHT);
            }
            prediction = learner.getVotesForInstance(instToActuallyPredict);
            if ((prediction.length > outlierLabel) && (prediction[outlierLabel] > (1.0 / prediction.length))) {
                this.pendingFinalLabelInstQueue.add(new TimeBoxedInstance(i,this.instancesProcessed, this.labelDeadlineOption.getValue(), prediction)); // Delay accuracy metrics until stale time
            } else {
                evaluator.addResult(instToActuallyPredict, prediction); // Not an outlier, so treat it like normal
                this.cm.add(weka.core.Utils.maxIndex(prediction),i.classValue());
            }
        }// end for
        
        assert this.pendingFinalLabelInstQueue.size() < (this.labelDeadlineOption.getValue() + 1) : "Cache 'pendingFinalLabelInstQueue' is larger than designed.";
        return ret;
    } //end test()

    /**
     *
     *
     * @return instances used for training
     */
    private int train() {
        this.monitor.setCurrentActivityDescription((this.inWarmupPhase) ? "Warmup Training" : "Online Training");
        int ret = 0;
        while (!this.latentTrainingInstQueue.isEmpty() && this.latentTrainingInstQueue.peek().deadline <= this.instancesProcessed) {
            Instance x = this.latentTrainingInstQueue.pop().inst;
            if (x.weight() > 0.0 || this.sendZeroWeightsOption.isSet()) {
                if (!x.classIsMissing()) {
                    learner.trainOnInstance(x);
                    this.knownLabels[(int) x.classValue()] += x.weight();
                    ret++;
                }
            }
        }
        assert this.latentTrainingInstQueue.size() < (this.trainingTimeDelayOption.getValue() + 1) : "Cache 'latentTrainingInstQueue' is larger than designed.";
        return ret;
    }

    /**
     * Test memory conditions
     *
     * @param monitor Task monitor for UI reporting
     * @param learningCurve for graphing
     * @param evaluateStartTime for timing 
     * @return true if we should continue processing
     */
    private boolean memoryTesting(TaskMonitor monitor, LearningCurve learningCurve, long evaluateStartTime) {
        this.monitor.setCurrentActivityDescription("Memory Housekeeping");
        if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
            if (monitor.taskShouldAbort()) {
                return false;
            }
            long estimatedRemainingInstances = stream.estimatedRemainingInstances();
            if (this.instanceLimitOption.getValue() > 0) {
                long maxRemaining = this.instanceLimitOption.getValue() - instancesProcessed;
                if ((estimatedRemainingInstances < 0)
                        || (maxRemaining < estimatedRemainingInstances)) {
                    estimatedRemainingInstances = maxRemaining;
                }
            }
            monitor.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0
                                                       ? -1.0
                                                       : (double) instancesProcessed / (double) (instancesProcessed + estimatedRemainingInstances));
            if (monitor.resultPreviewRequested()) {
                monitor.setLatestResultPreview(learningCurve.copy());
            }
            this.secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(TimingUtils
                    .getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
        }
        return true;
    }

    /**
     * Reset variables
     */
    private void reset() {
        if (knownLabels != null) { Arrays.fill(knownLabels,0);}
        this.stream = (InstanceStream) getPreparedClassOption(this.streamOption);
        this.learner = (Classifier) getPreparedClassOption(this.learnerOption);
        this.learner.setModelContext(stream.getHeader());
        this.evaluator = (ClassificationPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        this.pendingFinalLabelInstQueue.clear();
        this.latentTrainingInstQueue.clear();
        this.instancesProcessed = 0;
        this.secondsElapsed = 0;
        this.dumpFile = this.dumpFileOption.getFile();
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
            } catch (FileNotFoundException ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }
    }
    
        /**
     * 
     * @return purpose description of the class
     */
    @Override
    public String getPurposeString() {
        String compileDateStr = "(unknown)";
        try {
            File jarFile = new File(this.getClass().getProtectionDomain().getCodeSource().getLocation().toURI());
            Date compileDate = new Date(jarFile.lastModified());
            compileDateStr = compileDate.toString();
        } catch (URISyntaxException e) { }
        String ret = "Evaluates a classifier on a stream by testing then training with chunks of data in sequence. (build:" + compileDateStr + ")";
        return ret;
    }
}
