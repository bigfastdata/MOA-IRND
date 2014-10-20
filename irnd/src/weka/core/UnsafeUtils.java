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

package weka.core;

import java.lang.reflect.Field;

/**
 * Work-around for over-encapsulation of the WEKA classes
 * Note that in general this approach is a bad idea, but in this case it was too obnoxious to use inheritance, composition,
 * or other approaches due to the abundance of encapsulated uses of the WEKA Attribute objects.
 * @author bparker
 */
public class UnsafeUtils {
    
    /**
     * There REALLY REALLY should be a way to modify range of attributes for streaming use cases...
     * @param a Attribute to hack
     * @param lowerBoundValue lower limit of range
     * @param upperBoundValue upper limit of range
     * @return true if the JVM permits this hack, false if we fail
     */
    static public boolean setAttributeRange(Attribute a, double lowerBoundValue, double upperBoundValue) {
        SecurityManager sm = System.getSecurityManager();
        if (sm != null) {
            System.setSecurityManager(null); // Did I mention this utility class is unsafe?
        }
        try {
            // Now we can modify the Weka::Attribute fields we need to...
             Field lowerBoundField = a.getClass().getDeclaredField("m_LowerBound");
             lowerBoundField.setAccessible(true);
             lowerBoundField.setDouble(a, lowerBoundValue);
             Field upperBoundField = a.getClass().getDeclaredField("m_UpperBound");
             upperBoundField.setAccessible(true);
             upperBoundField.setDouble(a, upperBoundValue);
        } catch(NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException e) {
            System.err.println("Security setting prevents member modification via reflection.");
            return false;
        }
        // Be a decent steward of the JVM and at least put things back the way you found them...
        System.setSecurityManager(sm);
        return true;
    }
}
