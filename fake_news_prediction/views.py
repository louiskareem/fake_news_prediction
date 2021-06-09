import pickle
from django.shortcuts import render

nb_tfidf_model = pickle.load(open('fake_news_prediction/nb_tfidfvec_pred_v3.pkl', 'rb'))

def home(request):
    return render(request, 'index.html')


def predict(request):
    text_processing = pickle.load(open('fake_news_prediction/processed_data_v2.pkl', 'rb'))
    # Get the text
    djtext = request.POST.get('text')

    if djtext == '':
        prediction = "No Text Provided please try again"
    else:
        tfidf_vector = text_processing.transform([djtext])
        prediction = nb_tfidf_model.predict(tfidf_vector)
        prediction = prediction[0]

    params = {'Prediction': prediction}
    return render(request, 'result.html', params)
