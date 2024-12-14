from django.shortcuts import render
from .forms import TextForm
from .metin_analiz import ozellikleri_cikar_ve_puanla, duygu_analizi_modeli_olustur
from tensorflow.keras.preprocessing.text import Tokenizer

def analyze_text(request):
    result = None
    if request.method == 'POST':
        form = TextForm(request.POST)  # Formu al
        if form.is_valid():
            text = form.cleaned_data['text']  # Kullanıcının girdiği makale metnini al
            
            # Tokenizer ve modelin oluşturulması
            vocab_size = 10000
            embedding_dim = 100
            max_length = 100

            tokenizer = Tokenizer(num_words=vocab_size)
            tokenizer.fit_on_texts([text])
            model = duygu_analizi_modeli_olustur(vocab_size, embedding_dim, max_length)

            # Özelliklerin çıkarılması ve puanlama
            result = ozellikleri_cikar_ve_puanla(text, model, tokenizer, max_length)

    else:
        form = TextForm()

    return render(request, 'home.html', {'form': form, 'result': result})
