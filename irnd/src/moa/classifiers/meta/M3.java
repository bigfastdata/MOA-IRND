/*
 *    SluiceBox.java
 *    Copyright (C) 2013 Brandon S. Parker
 *    @author Brandon S. Parker (brandon.parker@utdallas.edu)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 *    
 */
package moa.classifiers.meta;

import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Random;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.functions.Perceptron;
import moa.classifiers.novelClass.AbstractNovelClassClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.StringUtils;
import moa.options.ClassOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 * M3.java
 *
 * This class was originally designed for use by the RandomMixedNovelDriftGenerator for MOA as part of Brandon Parker's
 * Dissertation work.
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 1 $
 */
public class M3 extends AbstractNovelClassClassifier {

    private static final long serialVersionUID = 1L;

    public FloatOption learningRateAlphaOption = new FloatOption("learningRateAlpha", 'r',
                                                                 "Learning Rate",
                                                                 0.51, 0.0, 1.0);

    public FloatOption pruneWeightEpsilonOption = new FloatOption("pruneWeightEpsilon", 'p',
                                                                  "If subordinate learner's weight drops below this value, it is replaced",
                                                                  0.125, 0.0, 0.5);

    public IntOption recoveryCacheSizeOption = new IntOption("recoveryCacheSize", 'k',
                                                             "Number instances to cache for sub-classifier recovery",
                                                             10, 1, 500);

    public IntOption numberOfEachBaseLearnersOption = new IntOption("numberOfEachBaseLearner", 'n',
                                                                    "Number of nominal-valued learners to use in ensemble",
                                                                    1, 0, 50);
    
    public ClassOption baseLearner1Option = new ClassOption("baseLearner1", '1',
                                                            "Base learner type 1", Classifier.class, "bayes.NaiveBayes");

    public ClassOption baseLearner2Option = new ClassOption("baseLearner2", '2',
                                                            "Base learner type 2", Classifier.class, "bayes.NaiveBayes");
    
    public ClassOption baseLearner3Option = new ClassOption("baseLearner3", '3',
                                                            "Base learner type 3", Classifier.class, "trees.HoeffdingTree");
    
    public ClassOption baseLearner4Option = new ClassOption("baseLearner4", '4',
                                                            "Base learner type 4", Classifier.class, "trees.HoeffdingTree -g 100");
    
    public ClassOption baseLearner5Option = new ClassOption("baseLearner5", '5',
                                                            "Base learner type 5", Classifier.class, "trees.HoeffdingTree -g 50");
    
    public ClassOption baseLearner6Option = new ClassOption("baseLearner6", '6',
                                                            "Base learner type 6", Classifier.class, "trees.HoeffdingTree -g 25");
    
    protected class EnsembleMemberMetrics {

        protected double defaultWeight = 0.5;
        protected double weight = 0.0;
        protected int numResets = 0;
        protected double weightSum = 0.0;
        protected int weightCount = 0;
        protected int updatesSinceLastReset = 0;

        EnsembleMemberMetrics(double w) {
            defaultWeight = w;
        }

        public double getWeight() {
            return weight;
        }

        public double getAverage() {
            return (weightCount > 0) ? this.weightSum / (double) weightCount : weight;
        }

        public int getResets() {
            return numResets;
        }

        public int getUpdatesSinceReset() {
            return updatesSinceLastReset;
        }

        public void setWeight(double w) {
            weight = w;
            weightCount++;
            weightSum += w;
            updatesSinceLastReset++;
        }

        public void reset() {
            numResets++;
            updatesSinceLastReset = 0;
            if (defaultWeight < 0 && weightCount > 0) {
                weight = this.weightSum / (double) weightCount;
            } else {
                weight = defaultWeight;
            }
        }

        public void reset(double w) {
            numResets++;
            updatesSinceLastReset = 0;
            weight = w;
            weightCount++;
            weightSum += w;
        }

        public void clear() {
            weight = defaultWeight;
            numResets = 0;
            weightSum = 0.0;
            weightCount = 0;
        }
    }
    /**
     *
     */
    protected Map<Classifier, EnsembleMemberMetrics> subordinateClassifiers = new HashMap<>();

    /**
     * Number of Naive Bayes sub-classifiers in use
     */
    protected int numNB = 0;
    /**
     * Number of perceptron sub-classifiers in use
     */
    protected int numPerceptrons = 0;
    /**
     * True once initialization has been executed (e.g. reset)
     */
    protected boolean reset = true;

    /**
     * Temporary cache used for retraining
     */
    protected Deque<Instance> recoveryCache;

    /**
     * Track the number of times a sub-classifer is too weak and thus reset
     */
    protected int tradeCounts = 0;

    @Override
    public void prepareForUseImpl(TaskMonitor mon, ObjectRepository repo) {
        // 1.) Basic setups
        this.reset = true;
        this.tradeCounts = 0;
        this.classifierRandom = new Random(42);
        mon.setCurrentActivity("Initializing Ensemble...", -1);
        recoveryCache = new LinkedList<>();

        // 2.) Initialize all subordinate Classifiers
        // 2.1) Base Classifier #1
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearner1Option);
        baseLearner.resetLearning();
        for (int i = 0; i < this.numberOfEachBaseLearnersOption.getValue(); i++) {
             this.subordinateClassifiers.put(baseLearner.copy(), new EnsembleMemberMetrics(this.classifierRandom.nextDouble()));
        }
        // 2.2) Base Classifier #2
        baseLearner = (Classifier) getPreparedClassOption(this.baseLearner2Option);
        baseLearner.resetLearning();
        for (int i = 0; i < this.numberOfEachBaseLearnersOption.getValue(); i++) {
             this.subordinateClassifiers.put(baseLearner.copy(), new EnsembleMemberMetrics(this.classifierRandom.nextDouble()));
        }
        // 2.3) Base Classifier #3
        baseLearner = (Classifier) getPreparedClassOption(this.baseLearner3Option);
        baseLearner.resetLearning();
        for (int i = 0; i < this.numberOfEachBaseLearnersOption.getValue(); i++) {
             this.subordinateClassifiers.put(baseLearner.copy(), new EnsembleMemberMetrics(this.classifierRandom.nextDouble()));
        }
        // 2.4) Base Classifier #4
        baseLearner = (Classifier) getPreparedClassOption(this.baseLearner4Option);
        baseLearner.resetLearning();
        for (int i = 0; i < this.numberOfEachBaseLearnersOption.getValue(); i++) {
             this.subordinateClassifiers.put(baseLearner.copy(), new EnsembleMemberMetrics(this.classifierRandom.nextDouble()));
        }
        // 2.5) Base Classifier #5
        baseLearner = (Classifier) getPreparedClassOption(this.baseLearner5Option);
        baseLearner.resetLearning();
        for (int i = 0; i < this.numberOfEachBaseLearnersOption.getValue(); i++) {
             this.subordinateClassifiers.put(baseLearner.copy(), new EnsembleMemberMetrics(this.classifierRandom.nextDouble()));
        }
        // 2.6) Base Classifier #6
        baseLearner = (Classifier) getPreparedClassOption(this.baseLearner6Option);
        baseLearner.resetLearning();
        for (int i = 0; i < this.numberOfEachBaseLearnersOption.getValue(); i++) {
             this.subordinateClassifiers.put(baseLearner.copy(), new EnsembleMemberMetrics(this.classifierRandom.nextDouble()));
        }

        if (mon.taskShouldAbort()) {
            return;
        }

        // 3.) Finish up
        super.prepareForUseImpl(mon, repo);
        //mon.setCurrentActivityFractionComplete(1.0);
        mon.setCurrentActivity("Initializing Complete...", -1);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (reset) {
            reset = false;
        }
        if (inst.weight() <= weka.core.Utils.SMALL) {
            return;
        }
        // 1.) Manage recovery cache
        final DenseInstance pseudoPoint = new DenseInstance((this.classifierRandom.nextDouble() + 0.5) / this.recoveryCacheSizeOption.getValue(), inst.toDoubleArray());
        pseudoPoint.setDataset(inst.dataset());
        this.recoveryCache.addFirst(pseudoPoint);
        if (this.recoveryCache.size() > this.recoveryCacheSizeOption.getValue()) {
            this.recoveryCache.removeLast();
        }
        final double alpha = this.learningRateAlphaOption.getValue();
        // 2.) Train the sub-classifiers
        for (Classifier c : this.subordinateClassifiers.keySet()) {
            final DenseInstance tmpPt = new DenseInstance((this.classifierRandom.nextDouble() + 0.5) / this.recoveryCacheSizeOption.getValue(), inst.toDoubleArray());
            tmpPt.setDataset(inst.dataset());
            final double pTrain = c.correctlyClassifies(inst) ? 1.0 : (1.0 - this.subordinateClassifiers.get(c).getWeight());
            tmpPt.setWeight(pTrain);
            c.trainOnInstance(tmpPt);
            double updatedWeight = this.subordinateClassifiers.get(c).getWeight() * alpha;// * pseudoPoint.weight());
            if (c.correctlyClassifies(inst)) {
                updatedWeight += (1.0 - alpha);// * pseudoPoint.weight();}
            }
            this.subordinateClassifiers.get(c).setWeight(updatedWeight);
        }
        // 3.) Manage sub-ordinate classifiers, resetting poor performers
        tradeOutBadModels();
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {

        DoubleVector ret = new DoubleVector();
        for (Classifier c : this.subordinateClassifiers.keySet()) {
            double weight = this.subordinateClassifiers.get(c).getWeight();
            //ret.addValues(c.getVotesForInstance(inst));
            DoubleVector subVote = new DoubleVector(c.getVotesForInstance(inst));
            subVote.scaleValues(weight);
            ret.addValues(subVote);
        }

        // Todo: use a small ANN to determine when we are correct.
        return ret.getArrayRef();
    }

    @Override
    public void resetLearningImpl() {
        for (Classifier c : this.subordinateClassifiers.keySet()) {
            c.resetLearning();
            this.subordinateClassifiers.get(c).clear();
        }
        this.recoveryCache.clear();
        this.reset = true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        StringUtils.appendIndented(out, indent, "M3 Online Streaming Ensemble using Reinforcement Learning for weights with " + this.subordinateClassifiers.size() + " members:");
        StringUtils.appendNewline(out);
        for (Classifier c : this.subordinateClassifiers.keySet()) {
            c.getDescription(out, indent);
        }
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public Classifier[] getSubClassifiers() {
        Classifier[] ret = new Classifier[this.subordinateClassifiers.size()];
        int i = 0;
        for (Classifier c : this.subordinateClassifiers.keySet()) {
            ret[i++] = c;
        }
        return ret;
    }

    /**
     * If an ensemble weight is below the prune level AND has not been recently reset (avoid thrashing), then reset it
     *
     * @return number of ensemble members reset
     */
    protected int tradeOutBadModels() {
        int ret = 0;
        double avgWeight = 0.0;
        double maxWeight = 0;
        for (EnsembleMemberMetrics v : this.subordinateClassifiers.values()) {
            avgWeight += v.getWeight();
            maxWeight = Math.max(v.weight, v.getWeight());
        }
        avgWeight /= this.subordinateClassifiers.size();

        for (Classifier c : this.subordinateClassifiers.keySet()) {
            if (this.subordinateClassifiers.get(c).getWeight() < this.pruneWeightEpsilonOption.getValue()
                    && this.subordinateClassifiers.get(c).getUpdatesSinceReset() >= this.recoveryCache.size()) {

                this.subordinateClassifiers.get(c).setWeight(0); // see how well everyone does without me first
                double errorFunctionSum = 0;
                double weightSum = 0;

                for (Instance x : this.recoveryCache) {
                    if (!this.correctlyClassifies(x)) {
                        errorFunctionSum += x.weight();
                    }
                }

                for (Instance x : this.recoveryCache) {
                    double newWeight = x.weight();
                    if (this.correctlyClassifies(x)) {
                        newWeight *= errorFunctionSum / (1.0 - errorFunctionSum);
                        if (Double.isNaN(newWeight)) {
                            newWeight = weka.core.Utils.SMALL;
                        }
                        x.setWeight(newWeight);
                    }
                    weightSum += newWeight;
                }

                for (Instance x : this.recoveryCache) {
                    x.setWeight(x.weight() / weightSum);
                }
                c.resetLearning();
                this.tradeCounts++;
                this.subordinateClassifiers.get(c).reset(avgWeight * 0.25 + maxWeight * 0.75);
                for (Instance x : this.recoveryCache) {
                    if ((classifierRandom.nextDouble() / this.recoveryCache.size()) < x.weight()) {
                        c.trainOnInstance(x);
                    }
                }
                ret++;
            }
        }
        return ret;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        double minW = Double.MAX_VALUE;
        double maxW = Double.MIN_VALUE;
        double avgW = 0;
        for (Classifier c : this.subordinateClassifiers.keySet()) {
            double w = this.subordinateClassifiers.get(c).getWeight();
            minW = Math.min(w, minW);
            maxW = Math.max(w, maxW);
            avgW += w;
        }
        avgW /= weka.core.Utils.SMALL + this.subordinateClassifiers.size();
        Measurement[] ret = new Measurement[]{
            new Measurement("MinWeight",
                            minW),
            new Measurement("MaxWeight",
                            maxW),
            new Measurement("AvgWeight",
                            avgW),
            new Measurement("NumMembers",
                            this.subordinateClassifiers.size()),
            new Measurement("ModelResetCount",
                            this.tradeCounts)
        };
        tradeCounts = 0;
        return ret;
    }

    /**
     * The SizeOfAgent method returns a value or -1 many times, so this override assures at least some estimate
     * using intrinsic knowledge of the object structure.
     *
     * @return Estimated numbed of bytes used by this object for data
     */
    @Override
    public int measureByteSize() {
        int ret = super.measureByteSize();
        if (ret <= 0) {
            ret = 24 + (12 * 3 + 32) * subordinateClassifiers.size();
            if (!recoveryCache.isEmpty()) {
                ret += recoveryCache.getFirst().numAttributes() * 8 * recoveryCache.size();
            }
        }
        return ret;
    }
}
