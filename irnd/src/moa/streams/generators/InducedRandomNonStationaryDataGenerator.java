/*
 *    InducedRandomNonStationaryDataGenerator.java
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
package moa.streams.generators;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import weka.core.DenseInstance;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;

import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;
import weka.core.Attribute;
import weka.core.Instances;

/**
 * Stream generator that facilitates concept and feature drift, as well as novel class emulation.
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 1 $
 */
public class InducedRandomNonStationaryDataGenerator extends AbstractOptionHandler implements
        InstanceStream {

    @Override
    public String getPurposeString() {
        return "Generates a dynamic data stream with non-stationary distributions for tunable value and label noise, Concept evolution, feature evolution, and concept drift.";
    }

    private static final long serialVersionUID = 1L;

    public IntOption modelRandomSeedOption = new IntOption("randomSeed", 's',
            "Seed for random generation of model.", 1, 1, Integer.MAX_VALUE);

    public IntOption numClassesOption = new IntOption("numClasses", 'C',
            "The total number of classes to generate in all.", 10, 2, Integer.MAX_VALUE);

    public IntOption numNominalAttributesOption = new IntOption("numNominalAttributes", 'o',
            "The number of nominal (non-ordinal) attributes.", 0, 0, Integer.MAX_VALUE);

    public IntOption numNumericAttributesOption = new IntOption("numNumericAttributes", 'n',
            "The number of numeric (real 0.0-1.0) attributes", 10, 0, Integer.MAX_VALUE);

    public FloatOption percentActiveClassesOption = new FloatOption("percentActiveClasses", 'P',
            "Fraction of classes active at any given time.  (0.0-100.0)", 30.0, 0.10, 100.00);

    public FloatOption percentActiveAttributesOption = new FloatOption("percentActiveAttributes", 'p',
            "Fraction of features active at any given time.  (0.0-100.0)", 100.0, 0.10, 100.00);
    
    public FloatOption classActivationProbabilityOption = new FloatOption("classActivationProbability", 'E',
            "Probability that active classes are exchanged  (0.0-100.0)", 0.001, 0.000, 100.000);

    public FloatOption attributeActivationProbabilityOption = new FloatOption("attributeActivationProbability", 'e',
            "Probability that active attributes/features are toggled (per concept, 0.0-100.0)", 0.000, 0.000, 100.000);

    public FloatOption attributeVelocityOption = new FloatOption("attributeVelocity", 'v',
            "Average drift in attribute loci (per concept, 0.0-100.0)", 0.05, 0.000, 100.000);

    public FloatOption attributeVelocityShiftProbabilityOption = new FloatOption("attributeVelocityShiftProbability", 'r',
            "The probability that an attributes velocity will shift  (0.0-100.0)", 0.01, 0, 100.0);
    
    public IntOption maxNumCentroidsPerClassOption = new IntOption("maxCentroidsPerClass", 'k',
            "The max number of centriods per attribute per label/class (i.e. size of GMM).", 5, 1, 50);

    public FloatOption attributeNoiseOption = new FloatOption("attributeNoise", 'A',
            "Noise factor added to attribute values. (0.0-100.0)", 5, 0.000, 100.000);
    
    public FloatOption labelNoiseOption = new FloatOption("labelNoise", 'L',
            "Probability the label of a class is incorrect  (0.0-100.0)", 0.000, 0.000, 100.000);
    
    protected InstancesHeader streamHeader;
    protected ArrayList<DriftingExemplarAttribute> featureSet;
    
    protected int numClasses = 2;
    protected int numActiveClasses = 1; 
    protected TreeSet<DriftingExemplarInstance> activeConcepts;
    protected TreeSet<DriftingExemplarInstance> inactiveConcepts;
    
    protected Random rng; // Random Number Generator

    @Override
    public void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {
        monitor.setCurrentActivity("Initializing concept exemplars...", -1.0);
        restart();
    }
    
    @Override
    public void restart() {
        this.rng = new Random(this.modelRandomSeedOption.getValue());
        this.numClasses = this.numClassesOption.getValue();
        this.numActiveClasses = (int) Math.ceil(this.percentActiveClassesOption.getValue()  / 100.0 * numClasses);
        this.activeConcepts = new TreeSet<>();
        this.inactiveConcepts = new TreeSet<>();
        generateHeader();
    }

    /**
     * Create the initial ARFF header section including the @RELATIONSHIP and @ATTRIBUTES section meta-data
     */
    protected void generateHeader() {
        int numNominalAtts = this.numNominalAttributesOption.getValue();
        int numNumericAtts = this.numNumericAttributesOption.getValue();
        int numTotalAtts = numNominalAtts + numNumericAtts;
        this.featureSet = new ArrayList<>(numTotalAtts + 1);
        int j = 1;
        for (int i = 0; i < numNumericAtts; i++) {
            DriftingExemplarAttribute newFeature = new DriftingExemplarAttribute("Attr_" + j + "_Num", j, rng);
            newFeature.setVelocity(this.attributeVelocityOption.getValue() / 100.0);
            featureSet.add(newFeature);
            j++;
        }
        for (int i = 0; i < numNominalAtts; i++) {
            DriftingExemplarAttribute newFeature = new DriftingExemplarAttribute("Attr_" + j + "_Nom", j, rng, generateBagOfWords(100));
            newFeature.setVelocity(this.attributeVelocityOption.getValue() / 100.0);
            featureSet.add(newFeature);
            j++;
        }
            
        ArrayList<String> labels = new ArrayList<>(this.numClasses);
        for(int i = 1; i <= this.numClasses; ++i) {
            Integer iVal = i;
            labels.add("Class_" + iVal.toString());
        }
        DriftingExemplarAttribute labelFeature;
        labelFeature = new DriftingExemplarAttribute("Class",this.featureSet.size(),rng,labels);
        labelFeature.setGMM(1,0);
        labelFeature.setVariance(0);
        labelFeature.setVelocity(0);
        featureSet.add(labelFeature);
        ArrayList<Attribute> attribs =  new ArrayList<>(this.featureSet.size());
        for (DriftingExemplarAttribute a: this.featureSet) {
            attribs.add(a.getAttribute());
        }
        DateFormat iso8601FormatString = new SimpleDateFormat("yyyyMMddHHmm");
        //iso8601FormatString.setTimeZone(TimeZone.getTimeZone("UTC"));
        String relationshipName = "InducedRandomNovelDrift" + iso8601FormatString.format(new Date());
        streamHeader = new InstancesHeader(new Instances(relationshipName,attribs,1));
        streamHeader.setClassIndex(this.featureSet.size() - 1);
        
        //Create initial exemplars/centroid for concept classes
        
        for(int i = 0; i < numClasses; ++i) {
            DriftingExemplarInstance c = new DriftingExemplarInstance(
                    featureSet, // Attribute mapping
                    i,      // Instance ID
                    (int) (numTotalAtts * this.percentActiveAttributesOption.getValue()  / 100.0),   // Num Active Attributes
                    this.attributeActivationProbabilityOption.getValue()  / 100.0,                   // P(change)
                    this.attributeVelocityOption.getValue()  / 100.0,                                // attribute velocity
                    this.attributeVelocityShiftProbabilityOption.getValue()  / 100.0,                // p(change(velocity)
                    this.attributeNoiseOption.getValue()  / 100.0,                                   // variance/noise
                    this.maxNumCentroidsPerClassOption.getValue(),                          // GMM size
                    this.rng);                                                              // shared Random Number Generator
            c.setDataset(streamHeader);
            if (i <= this.numActiveClasses) {
                this.activeConcepts.add(c);
            }
            else {
                this.inactiveConcepts.add(c);
            }
        }
        if (this.activeConcepts.isEmpty())
        {
            System.err.println("No concepts created!");
        }
    }
    
    @Override
    public Instance nextInstance() {
        // 1.) Pick a concept at weighted random
        DriftingExemplarInstance[] activeConceptsArray = activeConcepts.toArray(new DriftingExemplarInstance[1]);
        int nextConceptIdx = rng.nextInt(activeConceptsArray.length);

        // 2.) Pull the sampled Instance from the concept
        DenseInstance candidateInstance = activeConceptsArray[nextConceptIdx].nextInstance();
        Instance inst = new DenseInstance(candidateInstance);
        inst.setDataset(getHeader());
        
        // 3.) Add label noise
        if (rng.nextDouble() < this.labelNoiseOption.getValue()  / 100.0)
        {
            inst.setClassValue(rng.nextInt(numClasses));
        }
        
        // 4.) Shift active concepts
        if (!inactiveConcepts.isEmpty() && rng.nextDouble() < this.classActivationProbabilityOption.getValue()  / 100.0)
        {
            DriftingExemplarInstance[] inactiveConceptsArray = inactiveConcepts.toArray(new DriftingExemplarInstance[1]);
            int idxToActivate   = rng.nextInt(inactiveConceptsArray.length);
            int idxToDeactivate = rng.nextInt(activeConceptsArray.length);
            activeConcepts.add(inactiveConceptsArray[idxToActivate]);
            activeConcepts.remove(activeConceptsArray[idxToDeactivate]);
            inactiveConcepts.add(activeConceptsArray[idxToDeactivate]);
            inactiveConcepts.remove(inactiveConceptsArray[idxToActivate]);
        }
        
        //5.) Return new instance
        return inst;
    }

    /**
     * Creates a random bag of words for use by nominal attributes
     * @param size number of random words to create. Size of each word will be at least 4 char, but will scale to qty of words
     * @return ordered List of words created
     */
    protected List<String> generateBagOfWords(int size) {
        int idealWordLength = (int) Math.max(4,Math.log(size)/Math.log(13));
       
        String word;
        HashSet<String> bagOfWords = new HashSet<>(size);
        while(bagOfWords.size() < size)
        {
            word = "";
            for(int i = 0; i < idealWordLength; ++i)
            {
                word = word + (char) (this.rng.nextInt(26) + 97);
            }
            bagOfWords.add(word);
        }
        return new ArrayList<>(bagOfWords);
    }

    @Override
    public InstancesHeader getHeader() {
        return this.streamHeader;
    }

    @Override
    public long estimatedRemainingInstances() {
        return -1;
    }

    @Override
    public boolean hasMoreInstances() {
        return true;
    }

    @Override
    public boolean isRestartable() {
        return true;
    }
    
    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // TODO Auto-generated method stub
    }

}
