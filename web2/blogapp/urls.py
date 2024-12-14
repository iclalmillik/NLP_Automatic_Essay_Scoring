from django.urls import path
from . import views  # views.py dosyasından fonksiyonları dahil ediyoruz

urlpatterns = [
    path('', views.analyze_text, name='analyze_text'),  # Anasayfayı analyze_text fonksiyonuna yönlendiriyoruz
]
