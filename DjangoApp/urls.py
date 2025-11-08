from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('reglog/details/', views.regLog_details, name='reglog_details'),
    path('reglog/atelier/', views.regLog_atelier, name='reglog_atelier'),
    path('reglog/tester/', views.regLog_tester, name='reglog_tester'),
    path('reglog/prediction/', views.regLog_prediction, name='reglog_prediction'),
    path('dectree/details/', views.dectree_details, name='dectree_details'),
    path('dectree/atelier/', views.dectree_atelier, name='dectree_atelier'),
    path('dectree/tester/', views.dectree_tester, name='dectree_tester'),
    path('dectree/prediction/', views.dectree_prediction, name='dectree_prediction'),
]

