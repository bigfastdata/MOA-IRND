/*
 * Copyright 2014 bparker.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package moa.core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.TreeMap;
import java.util.TreeSet;
import moa.MOAObject;

/**
 *
 * @author bparker
 */
public class MultiClassConfusionMatrix implements MOAObject {
    private static final long serialVersionUID = 1L;
    
    private final static String seperator = ":";
    protected TreeMap<String, Double> table = new TreeMap<>();
    protected TreeSet<Integer> knownLabels = new TreeSet<>();
    protected String name = "Confusion Matrix";
    
    public MultiClassConfusionMatrix() { }
    public MultiClassConfusionMatrix(String n) { this.name = n; }
    
    public void add(double prediction, double truth) {
        String key = makeKey(prediction, truth);
        double oldValue = 0;
        if (table.containsKey(key)) { oldValue = table.get(key); }
        table.put(key, oldValue + 1);
        this.knownLabels.add((int)prediction);
        this.knownLabels.add((int)truth);
    }
    
    public void writeCSV(String filename) {
        BufferedWriter writer = null;
        try {
            File ConfusionMatrixFile = new File(filename);
            writer = new BufferedWriter(new FileWriter(ConfusionMatrixFile));
            String csv = this.toString();
            writer.write(csv);
        } catch (IOException e) {
        } finally {
            try {
                // Close the writer regardless of what happens...
                if (writer != null) { 
                    writer.flush();
                    writer.close(); 
                }
            } catch (IOException e) { }
        }        
    }
    
    @Override
    public String toString() {
        String outputLine = this.name;
        for(int prediction : knownLabels) {
            outputLine += String.format(",%d", prediction);
        }
        outputLine += "\n";
        for(int prediction : knownLabels) {
            outputLine += String.format("%d", prediction);
            for (int truth : knownLabels) {
                String key = makeKey(prediction,truth);
                double value = (table.containsKey(key)) ? table.get(key) : 0;
                outputLine += String.format(",%f", value);
            }
            outputLine += "\n";
        }
        return outputLine;
    }
    
    private String makeKey(double prediction, double truth) {
        return String.format("%d%s%d",(int)prediction,seperator,(int) truth);
    }

    @Override
    public int measureByteSize() {
        return name.length() + (table.size() * (3 + 8)) + this.knownLabels.size() * 4;
    }

    @Override
    public MOAObject copy() {
        MultiClassConfusionMatrix ret = new MultiClassConfusionMatrix(name);
        ret.knownLabels = this.knownLabels;
        ret.table = this.table;
        return this;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // not needed
    }
}
