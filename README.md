# Proyecto ORO

Sistema de apoyo a decisiones intradía para XAU/USD usando machine learning, variables técnicas y eventos macroeconómicos.

## Objetivo
Desarrollar un MVP capaz de integrar datos de mercado y calendario macroeconómico para generar señales cuantitativas de apoyo a decisiones intradía sobre oro.

## Problema
Las decisiones intradía en XAU/USD están expuestas a ruido de mercado y alta sensibilidad a noticias macroeconómicas, lo que dificulta priorizar entradas y gestionar riesgo de forma objetiva.

## Solución propuesta
El proyecto integra datos OHLCV de XAU/USD con un calendario macroeconómico, construye variables técnicas y macro, entrena modelos predictivos base y evalúa señales mediante backtesting.

## Componentes del MVP
- Integración de datos OHLCV a 5 minutos
- Integración de calendario macroeconómico
- Variables técnicas: retornos, momentum, RSI, ATR, volatilidad y medias móviles
- Variables macro: minutos al evento, impacto y eventos USD/FED/inflación/empleo
- Modelos base: Logistic Regression y Random Forest
- Generación de probabilidades, señales y backtesting

## Resultados principales
- Conjunto de datos modelable final: 12.741 observaciones y 43 variables de entrada
- Logistic Regression superó a Random Forest
- Logistic Regression: Accuracy 0.526 y AUC 0.532
- Random Forest: Accuracy 0.503 y AUC 0.513
- El MVP aún no alcanza rentabilidad consistente
- El modelo logístico redujo pérdidas frente a la referencia del mercado

## Archivos del repositorio
- `gold_ai_trading.py`: script principal
- `presentacion_proyecto_oro.pptx`: presentación final
- `presentacion_proyecto_oro.pdf`: versión PDF
- `resumen_modelos_v2.csv`: resumen de modelos
- `resumen_backtest_v2.csv`: resumen de backtest

## Próximos pasos
- Ampliar la ventana histórica de datos
- Optimizar thresholds de entrada
- Redefinir el target
- Incorporar filtro de tendencia
- Realizar validación walk-forward

## Autor
Franklin Patricio  
Universidad de Chile  
Diplomado en Ciencia de Datos para las Finanzas
