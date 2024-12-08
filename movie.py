import pandas as pd
import numpy as np
import spacy
import cv2
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re

class GenderStereotypeAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.role_classifier = None
        self.professional_terms = self._load_professional_terms()
        
    def _load_professional_terms(self):
        """Load dictionary of professional terms and their gender associations"""
        return {
            'masculine_coded': ['boss', 'chairman', 'businessman', 'salesman'],
            'feminine_coded': ['nurse', 'teacher', 'secretary', 'hostess'],
            'neutral': ['doctor', 'engineer', 'artist', 'professional']
        }
        
    def analyze_script(self, script_text):
        """Analyze movie script for gender stereotypes"""
        doc = self.nlp(script_text)
        
        analysis_results = {
            'character_analysis': self._analyze_characters(doc),
            'role_distribution': self._analyze_roles(doc),
            'relationship_patterns': self._analyze_relationships(doc),
            'dialogue_patterns': self._analyze_dialogue(doc)
        }
        
        return analysis_results
    
    def _analyze_characters(self, doc):
        """Extract and analyze character representations"""
        characters = {}
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                if ent.text not in characters:
                    characters[ent.text] = {
                        'mentions': 1,
                        'professional_roles': [],
                        'relationships': [],
                        'actions': []
                    }
                else:
                    characters[ent.text]['mentions'] += 1
                
                # Analyze surrounding context
                self._analyze_character_context(ent, characters[ent.text])
        
        return characters
    
    def _analyze_character_context(self, entity, char_dict):
        """Analyze context around character mentions"""
        sent = entity.sent
        
        # Extract professional roles
        for token in sent:
            if token.text.lower() in self.professional_terms['masculine_coded'] + \
               self.professional_terms['feminine_coded'] + \
               self.professional_terms['neutral']:
                char_dict['professional_roles'].append(token.text)
                
        # Extract relationships
        relationship_patterns = ['daughter of', 'son of', 'wife of', 'husband of']
        for pattern in relationship_patterns:
            if pattern in sent.text.lower():
                char_dict['relationships'].append(pattern)
                
    def _analyze_roles(self, doc):
        """Analyze distribution of professional and social roles"""
        roles = {
            'male': {'professional': 0, 'relationship': 0, 'undefined': 0},
            'female': {'professional': 0, 'relationship': 0, 'undefined': 0}
        }
        
        # Implementation of role analysis
        return roles
    
    def analyze_visual_stereotypes(self, image_path):
        """Analyze movie posters for visual stereotypes"""
        image = cv2.imread(image_path)
        
        analysis_results = {
            'position_analysis': self._analyze_character_positions(image),
            'size_analysis': self._analyze_character_sizes(image),
            'pose_analysis': self._analyze_character_poses(image)
        }
        
        return analysis_results
    
    def generate_recommendations(self, analysis_results):
        """Generate recommendations based on analysis"""
        recommendations = {
            'script_improvements': self._generate_script_recommendations(analysis_results),
            'visual_improvements': self._generate_visual_recommendations(analysis_results),
            'character_development': self._generate_character_recommendations(analysis_results)
        }
        
        return recommendations
    
    def _generate_script_recommendations(self, analysis):
        """Generate specific script improvement recommendations"""
        recommendations = []
        
        # Analyze professional role distribution
        if analysis['role_distribution']['female']['professional'] < \
           analysis['role_distribution']['male']['professional']:
            recommendations.append({
                'category': 'Professional Representation',
                'issue': 'Unequal professional role distribution',
                'suggestion': 'Include more female characters with defined professional roles'
            })
            
        # Analyze character introductions
        for char, details in analysis['character_analysis'].items():
            if 'daughter of' in details['relationships'] or 'wife of' in details['relationships']:
                recommendations.append({
                    'category': 'Character Introduction',
                    'issue': f'Character {char} introduced through male relationship',
                    'suggestion': 'Introduce character through their own achievements/characteristics'
                })
                
        return recommendations

    def _generate_visual_recommendations(self, analysis):
        """Generate recommendations for visual representation"""
        recommendations = []
        
        # Analyze poster composition
        if analysis.get('visual_stereotypes'):
            for issue in analysis['visual_stereotypes']:
                recommendations.append({
                    'category': 'Visual Representation',
                    'issue': issue['type'],
                    'suggestion': issue['improvement']
                })
                
        return recommendations

class StereotypeDetectionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = RandomForestClassifier()
        
    def train(self, texts, labels):
        """Train the stereotype detection model"""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        
    def predict(self, text):
        """Predict stereotypes in new text"""
        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]
        
def main():
    # Initialize analyzer
    analyzer = GenderStereotypeAnalyzer()
    
    script_text = """
    Rohit is an aspiring singer who works as a salesman in a car showroom, 
    run by Malik. One day he meets Sonia Saxena, daughter of Mr. Saxena, 
    when he goes to deliver a car to her home as her birthday present.
    """
    
    # Analyze script
    analysis_results = analyzer.analyze_script(script_text)
    
    # Generate recommendations
    recommendations = analyzer.generate_recommendations(analysis_results)
    
    # Print recommendations
    print("\nRecommendations for Improving Gender Representation:")
    for category, recs in recommendations.items():
        print(f"\n{category.upper()}:")
        for rec in recs:
            print(f"- Issue: {rec['issue']}")
            print(f"  Suggestion: {rec['suggestion']}")

if __name__ == "__main__":
    main()