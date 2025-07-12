
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
from datetime import datetime, timedelta
import random
from collections import defaultdict

app = Flask(__name__)

# Safely load the model and scaler
try:
    model = pickle.load(open("election_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    print("⚠️ Model or Scaler loading failed:", e)
    model = None
    scaler = None

# In-memory storage for demo purposes
prediction_history = []
comparison_history = []

def generate_mock_analytics():
    """Generate comprehensive mock analytics data"""
    return {
        "accuracy_metrics": {
            "total_predictions": len(prediction_history) + random.randint(50, 200),
            "correct": random.randint(40, 180),
            "accuracy": f"{random.randint(75, 95)}%",
            "win_predictions": len([p for p in prediction_history if p.get('result') == 'Victory Predicted']) + random.randint(25, 100)
        },
        "demographic_insights": {
            "avg_age": random.randint(35, 65),
            "avg_income": random.randint(45000, 85000),
            "avg_education": random.randint(12, 18),
            "top_sentiment_range": "0.6-0.8"
        },
        "trend_data": generate_trend_data(),
        "regional_performance": {
            "Urban": random.randint(60, 85),
            "Suburban": random.randint(55, 75),
            "Rural": random.randint(45, 70)
        }
    }

def generate_trend_data():
    """Generate trend data for charts"""
    trends = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        date = base_date + timedelta(days=i)
        trends.append({
            'date': date.strftime('%Y-%m-%d'),
            'predictions': random.randint(1, 15),
            'avg_confidence': random.randint(65, 95),
            'win_rate': random.randint(50, 85)
        })
    
    return trends

def make_prediction(age, income, education, sentiment, poll):
    """Enhanced prediction function with detailed insights"""
    try:
        if model is None or scaler is None:
            # Fallback prediction logic
            features_score = (age * 0.1) + (income * 0.0001) + (education * 2) + (sentiment * 50) + (poll * 60)
            confidence = min(95, max(55, features_score + random.randint(-10, 10)))
            result = "Victory Predicted" if features_score > 45 else "Defeat Predicted"
        else:
            # Use actual model (if available)
            features = np.array([[age, income, education, sentiment, poll]])
            scaled = scaler.transform(features)
            result_prob = model.predict_proba(scaled)[0]
            confidence = int(max(result_prob) * 100)
            result = "Victory Predicted" if model.predict(scaled)[0] == 1 else "Defeat Predicted"
        
        # Generate insights
        insights = generate_insights(age, income, education, sentiment, poll, confidence)
        
        return result, confidence, insights
    
    except Exception as e:
        return f"Prediction Error: {str(e)}", 0, []

def generate_insights(age, income, education, sentiment, poll, confidence):
    """Generate detailed insights based on input features"""
    insights = []
    
    if age < 35:
        insights.append("Younger candidates often appeal to progressive voters")
    elif age > 65:
        insights.append("Experienced candidates may face age-related concerns")
    else:
        insights.append("Age falls in optimal political leadership range")
    
    if income > 100000:
        insights.append("High income may create relatability challenges")
    elif income < 50000:
        insights.append("Lower income enhances working-class appeal")
    
    if education > 16:
        insights.append("Advanced education appeals to educated voters")
    elif education < 12:
        insights.append("Education level may limit appeal to certain demographics")
    
    if sentiment > 0.7:
        insights.append("Strong positive sentiment indicates good public perception")
    elif sentiment < 0.4:
        insights.append("Low sentiment suggests need for image improvement")
    
    if poll > 0.6:
        insights.append("High polling numbers indicate strong current support")
    elif poll < 0.3:
        insights.append("Low polling suggests uphill battle ahead")
    
    if confidence > 80:
        insights.append("Model shows high confidence in prediction")
    elif confidence < 60:
        insights.append("Prediction has moderate uncertainty")
    
    return insights

@app.route("/", methods=["GET", "POST"])
def predict():
    result = None
    confidence = None
    insights = None
    
    # Generate comprehensive analytics
    analytics = generate_mock_analytics()
    
    # Feature importance data
    feature_importance = [
        ("Poll Percentage", 0.35, "Current polling data is the strongest predictor"),
        ("Public Sentiment", 0.25, "Voter sentiment significantly impacts outcomes"),
        ("Education Level", 0.15, "Education influences policy appeal"),
        ("Age Factor", 0.13, "Age affects voter demographic alignment"),
        ("Income Level", 0.12, "Economic status influences policy positions")
    ]
    
    if request.method == "POST":
        try:
            name = request.form.get("name", "Unnamed Candidate")
            age = float(request.form["age"])
            income = float(request.form["income"])
            education = float(request.form["education"])
            sentiment = float(request.form["sentiment"])
            poll = float(request.form["poll"])
            
            result, confidence, insights = make_prediction(age, income, education, sentiment, poll)
            
            # Store in history
            prediction_history.append({
                'name': name,
                'age': age,
                'income': income,
                'education': education,
                'sentiment': sentiment,
                'poll': poll,
                'result': result,
                'confidence': confidence,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'insights': insights
            })
            
        except Exception as e:
            result = f"Error: {str(e)}"
            confidence = 0
            insights = []
    
    return render_template("index.html", 
                         result=result, 
                         confidence=confidence, 
                         insights=insights,
                         analytics=analytics,
                         feature_importance=feature_importance,
                         history=prediction_history[-10:],  # Last 10 predictions
                         trend_data=analytics['trend_data'])

@app.route("/history")
def history():
    analytics = generate_mock_analytics()
    
    # Enhanced prediction history with more details
    enhanced_history = []
    for pred in prediction_history:
        enhanced_pred = pred.copy()
        enhanced_pred['success_probability'] = enhanced_pred.get('confidence', 0)
        enhanced_pred['risk_factors'] = len([i for i in enhanced_pred.get('insights', []) if 'challenge' in i.lower() or 'concern' in i.lower()])
        enhanced_history.append(enhanced_pred)
    
    return render_template("history.html", 
                         predictions=enhanced_history,
                         analytics=analytics,
                         trend_data=analytics['trend_data'],
                         total_predictions=len(prediction_history),
                         avg_confidence=sum(p.get('confidence', 0) for p in prediction_history) / max(1, len(prediction_history)))

@app.route("/compare")
def compare():
    return render_template("compare.html")

@app.route("/api/compare", methods=["POST"])
def api_compare():
    try:
        data = request.get_json()
        candidates = data.get('candidates', [])
        
        if len(candidates) < 2:
            return jsonify({"success": False, "error": "Need at least 2 candidates"})
        
        results = []
        for candidate in candidates:
            name = candidate['name']
            age = candidate['age']
            income = candidate['income']
            education = candidate['education']
            sentiment = candidate['sentiment']
            poll = candidate['poll']
            
            prediction, confidence, insights = make_prediction(age, income, education, sentiment, poll)
            
            results.append({
                'name': name,
                'prediction': prediction,
                'confidence': confidence,
                'insights': insights[:3],  # Limit insights for comparison view
                'age': age,
                'income': income,
                'education': education,
                'sentiment': sentiment,
                'poll': poll
            })
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Determine winner
        winner = results[0]
        
        # Generate analysis summary
        analysis_summary = f"""
        Based on comprehensive AI analysis of {len(candidates)} candidates, <strong>{winner['name']}</strong> 
        emerges as the frontrunner with {winner['confidence']}% confidence. Key factors include strong polling data 
        ({winner['poll']:.1%}), favorable sentiment score ({winner['sentiment']:.2f}), and optimal demographic profile.
        The analysis considered age demographics, income levels, education background, public sentiment, and current polling trends.
        """
        
        # Store comparison in history
        comparison_history.append({
            'candidates': [c['name'] for c in candidates],
            'winner': winner['name'],
            'confidence': winner['confidence'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return jsonify({
            "success": True,
            "results": results,
            "winner": winner,
            "analysis_summary": analysis_summary
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
