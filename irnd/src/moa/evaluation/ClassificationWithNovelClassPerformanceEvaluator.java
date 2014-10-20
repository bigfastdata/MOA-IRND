package moa.evaluation;

/*
 *    ClassificationWithNovelClassPerformanceEvaluator.java
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
 *    This class is derived from the ClassificationPerformanceEvaluator.java class which
 *    is also licensed under the GPL Version 3 and contains the notice:

 *    BasicClassificationPerformanceEvaluator.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 *  
 */


import java.util.Arrays;
import moa.classifiers.novelClass.AbstractNovelClassClassifier;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;


/**
 * Evaluates the classification AND novel class detection accuracy of classification test
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 1 $
 */
public class ClassificationWithNovelClassPerformanceEvaluator extends AbstractOptionHandler  implements ClassificationPerformanceEvaluator {
    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Evaluate classifier accuracy with added accuracy and metrics for novel class detection (key for MineClass, DXMiner, and SluiceBox).";
    }
      
    /**
     * Minimum value for novel label vote for declaration of novelty.
     */
//    public FloatOption thresholdOfNoveltyOption = new FloatOption("noveltyThreshold", 't',
//                                                     "Minimum value for novel label vote for declaration of novelty.",
//                                                     0.5, 0.0, 1.0);
    
    public IntOption observationsUntilNotNovelOption = new IntOption("observationsUntilNotNovel", 'n',
                                                     "Number of h(x) observations of a class to see before it is no longer considered novel",
                                                    50,1,Integer.MAX_VALUE);
    
    public IntOption maxUnobservationsUntilNotNovelOption = new IntOption("maxUnobservationsUntilNotNovel", 'N',
                                                     "Number of true label observations of a class to see before it is no longer considered novel",
                                                     1000,1,Integer.MAX_VALUE);
    
    public MultiChoiceOption outlierHandlingStrategyOption = new MultiChoiceOption("ooutlierHandlingStrategy", 'o',
                                                                                     "Set strategy for how to handle outliers.",
                                                                                     new String[]{"Evaluate Anyway", "Ignore Marked", "Ignore unvoted", "Ignore Pure", "Ignore Either","Treat as Novel"},
                                                                                     new String[]{"Evaluate Anyway", "Ignore Marked", "Ignore unvoted", "Ignore Pure", "Ignore Either","Treat as Novel"},
                                                                                     0);
    
    public final FlagOption goodIsGoodOption = new FlagOption("goodIsGoodNumber", 'G',
            "Select this if the Classifier is not a novel-class detector. If h(x) == y, then it will not be penalized on overall accuracy");
    
    protected double weightObserved;

    protected double weightCorrect;

    protected double[] columnKappa;

    protected double[] rowKappa;

    protected int[] observedLabels = null;
    protected int[] knownTrueLabels = null;
    
    protected int numClasses;

    private int novelClassLabel = 0;
    
    private int outlierLabel = 0;
    
    private double novelClassDetectionFalsePositive = 0;
    
    private double novelClassDetectionFalseNegative = 0;
    
    private double novelClassDetectionTruePositive = 0;
    
    private double novelClassDetectionTrueNegative = 0;
    
    private long numberOfInstancesSeen = 0;
    
    // Constants for "magic numbers"
    protected static final float  HASH_THRESHOLD_FOR_A_PRIORI_LENGTHS = 0.99999f;
    protected Instances header = null;
    
    @Override
    public void reset() {
        header = null;
        this.weightObserved = 0.0;
        this.weightCorrect = 0.0;
        this.numberOfInstancesSeen = 0;
        this.novelClassDetectionFalsePositive = 0;
        this.novelClassDetectionFalseNegative = 0;
        this.novelClassDetectionTruePositive = 0;
        this.novelClassDetectionTrueNegative = 0;        
    }

    /**
     * 
     * Note that for novel class testing, an addition class value is added to the known classes. T
     * This extra "Label" represents a prediction of "Novel Class". This approach allows for
     * algorithms that do not have novel class prediction capabilities to still function,
     * as this method first bounds checks to see if the prediction array includes the added label
     * 
     * @param inst instance under test
     * @param classVotes prediction table for this instance
     */
    @Override
    public void addResult(Instance inst, double[] classVotes) {
        if (header == null) { 
            header = AbstractNovelClassClassifier.augmentInstances(inst.dataset());
            this.novelClassLabel = header.classAttribute().indexOfValue(AbstractNovelClassClassifier.NOVEL_LABEL_STR);
            this.outlierLabel = header.classAttribute().indexOfValue(AbstractNovelClassClassifier.OUTLIER_LABEL_STR);
            this.rowKappa = new double[header.numClasses()];
            Arrays.fill(this.rowKappa,0.0);
            this.columnKappa = new double[header.numClasses()];
            Arrays.fill(this.columnKappa,0.0);
            this.knownTrueLabels = new int[header.numClasses()];
            Arrays.fill(knownTrueLabels,0);
            this.observedLabels = new int[header.numClasses()];
            Arrays.fill(observedLabels,0);
        }

        final int trueClass = (int) inst.classValue();
        if (classVotes == null) {
            this.knownTrueLabels[trueClass]++;
            return;
        }
        final double[] labelsOnlyVotes = Arrays.copyOf(classVotes, inst.numClasses());
        if (labelsOnlyVotes.length > this.novelClassLabel) {labelsOnlyVotes[novelClassLabel] = 0;}
        if (labelsOnlyVotes.length > this.outlierLabel) {labelsOnlyVotes[outlierLabel] = 0;}
        final double totalVoteQty = weka.core.Utils.sum(labelsOnlyVotes);
        final int predictedClass = weka.core.Utils.maxIndex(labelsOnlyVotes); // Don't count the special extended indexes for novel and outlier
        final boolean isMarkedOutlier = (weka.core.Utils.maxIndex(classVotes) == this.outlierLabel);
        
        if (predictedClass < inst.numClasses() && labelsOnlyVotes[predictedClass] > 0.0) { // Only if there is SOME vote (non-zero)
            this.observedLabels[predictedClass]++; // If we predict it, then it can't be novel!
        }
        //final boolean isTrueNovel = !(this.observedLabels[(int)trueClass] > observationsUntilNotNovelOption.getValue());
        boolean predictedNovel = ((classVotes.length > this.novelClassLabel) && (classVotes[this.novelClassLabel] > 0));// this.thresholdOfNoveltyOption.getValue()));
        
        final boolean isVoteOutlier = (totalVoteQty <= (weka.core.Utils.SMALL * 10.0));
        final boolean correctLabelPrediction = (predictedClass == trueClass);
        switch (this.outlierHandlingStrategyOption.getChosenIndex()) {
            case 0: // use anyway
                // keep on trucking... 
                break;
            case 1: // ignore marked
                if (isMarkedOutlier) {
                    return;
                }
                break;
            case 2: // ignore no vote
                if (isVoteOutlier) {
                    return;
                }
                break;
            case 3: // ignore iff marked AND no vote
                if (isVoteOutlier && isMarkedOutlier) {
                    return;
                }
                break;
            case 4: // ignore pure OR marked
                if (isVoteOutlier || isMarkedOutlier) {
                    return;
                }
                break;
            case 5: // mark as novel
                predictedNovel = predictedNovel || isMarkedOutlier;
                break;
            default:
                break;
        }
        this.numberOfInstancesSeen++;
        this.weightObserved += inst.weight();     // /!\ IS THIS RIGHT???
        //final boolean isTrueNovel = (this.knownTrueLabels[trueClass] < this.maxUnobservationsUntilNotNovelOption.getValue()) && (this.observedLabels[trueClass] < observationsUntilNotNovelOption.getValue());
        final boolean isTrueNovel = (this.knownTrueLabels[trueClass] < this.maxUnobservationsUntilNotNovelOption.getValue());
        // 8x different mutually exclusive options (i.e. 3-bits)
        if ((!predictedNovel) && (!isTrueNovel) && (correctLabelPrediction)) { // Should be most common
            this.novelClassDetectionTrueNegative++;
            this.weightCorrect++;
        } 
        if ((predictedNovel) && (isTrueNovel) && (correctLabelPrediction)) { // Rare if ever
            this.novelClassDetectionTruePositive++;
            this.weightCorrect++;
            assert false : "Paradox 1 - true novel, but predicted the right label";
        }
        if ((predictedNovel) && (!isTrueNovel) && (correctLabelPrediction)) { // Error due to overly restrictive models
            this.novelClassDetectionFalsePositive++;
            if (this.goodIsGoodOption.isSet()) {
                this.weightCorrect++;
            }
        }
        if ((!predictedNovel) && (isTrueNovel) && (correctLabelPrediction)) { // Should never happen?  Framework was wrong here, so TN
            this.novelClassDetectionTrueNegative++;
            this.weightCorrect++;
            assert false : "Paradox 2 - true novel, but predicted the right label";
        }
        if ((predictedNovel) && (isTrueNovel) && (!correctLabelPrediction)) { // Should be most common when x is novel
            this.novelClassDetectionTruePositive++;
            this.weightCorrect++;
        }
        if ((predictedNovel) && (!isTrueNovel) && (!correctLabelPrediction)) { // Probably an Outlier case
            this.novelClassDetectionFalsePositive++;
            if (this.outlierHandlingStrategyOption.getChosenIndex() > 0) {
                this.weightCorrect++;
            }
        }
        if ((!predictedNovel) && (isTrueNovel) && (!correctLabelPrediction)) { // NCD failure     FN
            this.novelClassDetectionFalseNegative++;
        }
        if ((!predictedNovel) && (!isTrueNovel) && (!correctLabelPrediction)) { // Correct NCD, but bad h(x) prediction
            this.novelClassDetectionTrueNegative++;
        }

        this.rowKappa[predictedClass]++;
        this.columnKappa[trueClass]++;
        this.knownTrueLabels[trueClass] += inst.weight();

    }

    /**
     * Return set of measurements for MOA reporting
     * @return measurements array
     */
    @Override
    public Measurement[] getPerformanceMeasurements() {
        double Fnew = (( this.numberOfInstancesSeen - (novelClassDetectionFalseNegative + novelClassDetectionTruePositive)) == 0 ) ? 0 : novelClassDetectionFalsePositive * 100.0 / ( this.numberOfInstancesSeen - (novelClassDetectionFalseNegative + novelClassDetectionTruePositive));
        double Mnew = ((novelClassDetectionFalseNegative + novelClassDetectionTruePositive) == 0) ? 0 : novelClassDetectionFalseNegative * 100.0 / (novelClassDetectionFalseNegative + novelClassDetectionTruePositive);
        int numPredictedLabels = 0;
        int numTrueLabels = 0;
        for (int i = 0; i < this.header.numClasses(); ++i) {
            if (this.knownTrueLabels[i] > 0) {numTrueLabels++;} 
            if (this.observedLabels[i] > 0)  {numPredictedLabels++;}
        }
        Measurement[] ret = new Measurement[]{
            new Measurement("classified instances",
            this.numberOfInstancesSeen),
            new Measurement("classifications correct (percent)",
            getFractionCorrectlyClassified() * 100.0),
            new Measurement("Kappa Statistic (percent)",
            getKappaStatistic() * 100.0),
            new Measurement("Correct Base Predictions (count)",
            weightCorrect),
            new Measurement("Novel Class TP (count)",
            novelClassDetectionTruePositive),
            new Measurement("Novel Class FP (count)",
            novelClassDetectionFalsePositive),
            new Measurement("Novel Class FN (count)",
            novelClassDetectionFalseNegative),
            new Measurement("Novel Class TN (count)",
            novelClassDetectionTrueNegative),
            new Measurement("Mnew",Mnew),
            new Measurement("Fnew",Fnew),
            new Measurement("NumTrueLabels",
            numTrueLabels),
            new Measurement("NumOutputLabels",
            numPredictedLabels)
        };
        numberOfInstancesSeen = 0;
        weightCorrect = 0;
        novelClassDetectionTruePositive = 0;
        novelClassDetectionFalsePositive = 0;
        novelClassDetectionFalseNegative = 0;
        novelClassDetectionTrueNegative = 0;
        Arrays.fill(this.rowKappa,0.0);
        Arrays.fill(this.columnKappa,0.0);
        return ret;
    }

    public double getTotalWeightObserved() {
        return this.weightObserved;
    }

     public double getFractionCorrectlyClassified() {
         // Base prediction with Novel Class prediction correction
        double correctCount = this.weightCorrect;
        return this.numberOfInstancesSeen > 0.0 ? (correctCount / this.numberOfInstancesSeen) : 0.0;
    }
    
    public double getFractionIncorrectlyClassified() {
        return 1.0 - getFractionCorrectlyClassified();
    }

    /**
     * todo: should weightObserved really be numberOfInstances?
     * @return Kappa metrics for reporting
     */
    public double getKappaStatistic() {
        if (this.weightObserved > 0.0) {
            double p0 = getFractionCorrectlyClassified();
            double pc = 0.0;
            for (int i = 0; i < this.numClasses; i++) {
                pc += (this.rowKappa[i] / this.numberOfInstancesSeen)
                        * (this.columnKappa[i] / this.numberOfInstancesSeen);
            }
            if (pc != 1.0) {
                return (p0 - pc) / (1.0 - pc);
            } else {
                return (p0 > pc) ? 1.0 : (p0 < pc) ? -1.0 : 0.0;
            }
        }
        return 0;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        Measurement.getMeasurementsDescription(getPerformanceMeasurements(),
                sb, indent);
    }

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        this.reset();
    }
}
