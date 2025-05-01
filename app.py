from flask import Flask, render_template, request, session
import os
from pdfminer.high_level import extract_text
import spacy
import matplotlib.pyplot as plt
from io import BytesIO

try:
    from transformers import pipeline
    import torch  # Ensure PyTorch is available
    text_generator = pipeline("text-generation", model="gpt2")  # Load GPT-2 model
except ImportError as e:
    text_generator = None
    print(f"Error: {e}. Please install the required libraries using 'pip install transformers torch'.")

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session handling

nlp = spacy.load("en_core_web_sm")  # Load spaCy model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ensure directories exist
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('static', exist_ok=True)

        # Get the uploaded file and job description
        file = request.files['resume']
        job_description = request.form['job_description']
        if file and job_description:
            # Save the file to a temporary location
            temp_file_path = os.path.join('uploads', file.filename)
            file.save(temp_file_path)

            # Extract text from the resume
            resume_text = extract_text(temp_file_path)

            # Analyze the resume and job description
            analysis_results, similarity_score, improvement_tips = analyze_resume(resume_text, job_description)

            # Generate AI-based interview tips
            ai_tips = generate_ai_tips(job_description, improvement_tips['missing_keywords'])

            # Generate a visualization
            chart_path = generate_visualization(similarity_score, analysis_results)

            # Store results in session temporarily
            session['analysis_results'] = analysis_results
            session['chart_path'] = chart_path
            session['improvement_tips'] = improvement_tips
            session['ai_tips'] = ai_tips

            # Render the results and clear session afterward
            response = render_template(
                'index.html',
                message='Analysis complete!',
                analysis_results=analysis_results,
                chart_path=chart_path,
                improvement_tips=improvement_tips,
                ai_tips=ai_tips
            )
            session.clear()  # Ensure session clearing does not interfere
            return response
    else:
        # Clear session data on first load
        session.clear()

    return render_template('index.html')

def analyze_resume(resume_text, job_description):
    # Process the resume and job description using spaCy
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description)

    # Calculate similarity score
    similarity_score = resume_doc.similarity(job_doc)

    # Extract keywords from job description
    job_keywords = [token.text for token in job_doc if token.is_alpha and not token.is_stop]

    # Check for keyword matches in the resume
    matched_keywords = [word for word in job_keywords if word.lower() in resume_text.lower()]
    missing_keywords = [word for word in job_keywords if word.lower() not in resume_text.lower()]

    # Generate analysis results
    analysis_results = {
        "similarity_score": round(similarity_score * 100, 2),
        "matched_keywords": matched_keywords,
        "total_keywords": len(job_keywords),
        "match_percentage": round(len(matched_keywords) / len(job_keywords) * 100, 2) if job_keywords else 0
    }

    # Generate improvement tips
    improvement_tips = {
        "missing_keywords": missing_keywords,
        "suggestion": generate_suggestions(missing_keywords)
    }

    return analysis_results, similarity_score, improvement_tips

def generate_suggestions(missing_keywords):
    if not missing_keywords:
        return "Your resume is a perfect match for the job description!"
    else:
        return f"Consider adding these keywords to your resume to improve the match: {', '.join(missing_keywords)}"

def generate_ai_tips(job_description, missing_keywords):
    """
    Generate AI-based interview tips using a pre-trained NLP model.
    """
    if not text_generator:
        return ["AI tips are unavailable. Please install 'transformers' and 'torch'."]

    if not missing_keywords:
        return ["Your resume is already a perfect match for the job description!"]

    # Generate a prompt for the AI model
    prompt = (
        f"The job description is: {job_description}\n"
        f"The following keywords are missing from the resume: {', '.join(missing_keywords)}.\n"
        "Provide actionable tips to improve the resume and prepare for the interview:"
    )

    # Use the text generation pipeline to generate tips
    ai_response = text_generator(prompt, max_length=100, num_return_sequences=1)
    tips = ai_response[0]['generated_text'].split('\n')  # Split the response into individual tips

    return tips

def generate_visualization(similarity_score, analysis_results):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart for similarity score
    axes[0].bar(["Similarity"], [similarity_score * 100], color='blue')
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percentage")
    axes[0].set_title("Resume-Job Description Similarity")

    # Pie chart for matched vs unmatched keywords
    matched = len(analysis_results["matched_keywords"])
    unmatched = analysis_results["total_keywords"] - matched
    axes[1].pie(
        [matched, unmatched],
        labels=["Matched Keywords", "Unmatched Keywords"],
        autopct='%1.1f%%',
        colors=['green', 'red'],
        startangle=90
    )
    axes[1].set_title("Keyword Match Distribution")

    # Save the chart to a BytesIO object
    chart_io = BytesIO()
    plt.tight_layout()
    plt.savefig(chart_io, format='png')
    chart_io.seek(0)
    chart_path = os.path.join('static', 'similarity_chart.png')
    with open(chart_path, 'wb') as f:
        f.write(chart_io.read())
    chart_io.close()

    return chart_path

if __name__ == '__main__':
    # Create the uploads and static directories if they don't exist
    for directory in ['uploads', 'static']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    app.run(debug=True)