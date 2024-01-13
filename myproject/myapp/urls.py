# myapp/urls.py
from django.urls import path
from .views import radar_chart_view, ajax_radar_chart_view

urlpatterns = [
    path('radar_chart/', radar_chart_view, name='radar_chart'),
    path('ajax/radar_chart/', ajax_radar_chart_view, name='ajax_radar_chart'),
]
