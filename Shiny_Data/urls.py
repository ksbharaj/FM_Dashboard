from django.urls import path
from . import views

urlpatterns = [
    path('radar_chart/', views.radar_chart_view, name='radar_chart'),
]