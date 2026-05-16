import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple

# =========================
# CONFIGURACIÓN DE LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# CONFIGURACIÓN
# =========================
CAPITAL_INICIAL = 1000
RIESGO_POR_OPERACION = 0.02   # 2%
TIEMPO_ENTRE_OPERACIONES = 60 # segundos
MAX_PERDIDAS_CONSECUTIVAS = 3
COMISION = 0.001  # 0.1%
SLIPPAGE = 0.002  # 0.2%

# =========================
# INDICADORES TÉCNICOS
# =========================
def ema(series, period=20):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    """MACD - Moving Average Convergence Divergence"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series, period=20, num_std=2):
    """Bollinger Bands"""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


# =========================
# GENERADOR DE DATOS (Simular datos reales)
# =========================
def obtener_datos(candles=100, volatility=0.02):
    """
    Genera datos de prueba realistas
    candles: número de velas
    volatility: volatilidad del precio
    """
    np.random.seed(42)
    returns = np.random.normal(0.001, volatility, candles)
    precios = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "close": precios,
        "timestamp": pd.date_range(start='2024-01-01', periods=candles, freq='1H')
    })
    return df


# =========================
# CLASE PARA EL BOT DE TRADING
# =========================
class TradingBot:
    def __init__(self, capital_inicial: float):
        self.capital = capital_inicial
        self.capital_inicial = capital_inicial
        self.posicion_abierta = False
        self.entrada_precio = 0
        self.entrada_tamanio = 0
        self.perdidas_consecutivas = 0
        self.operaciones = []
        self.historial_capital = [capital_inicial]
        logger.info(f"Bot inicializado con capital: ${capital_inicial}")

    def calcular_tamanio_posicion(self, precio_actual: float) -> float:
        """Calcula el tamaño de la posición basado en el riesgo"""
        riesgo = self.capital * RIESGO_POR_OPERACION
        tamanio = riesgo / precio_actual
        return tamanio

    def analizar_mercado(self, df: pd.DataFrame) -> str:
        """
        Genera señal de trading basada en múltiples indicadores
        Retorna: 'COMPRA', 'VENTA', 'NO TRADE'
        """
        if len(df) < 30:
            return 'NO TRADE'

        df = df.copy()
        
        # Calcular indicadores
        df['EMA20'] = ema(df['close'], 20)
        df['EMA50'] = ema(df['close'], 50)
        df['RSI14'] = rsi(df['close'], 14)
        df['MACD'], df['Signal'], df['Histogram'] = macd(df['close'])
        df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = bollinger_bands(df['close'])

        ultima = df.iloc[-1]
        anterior = df.iloc[-2]

        # Condiciones de COMPRA
        if (ultima['close'] > ultima['EMA20'] and 
            ultima['EMA20'] > ultima['EMA50'] and
            ultima['RSI14'] < 70 and
            ultima['Histogram'] > 0 and
            anterior['Histogram'] <= 0):
            logger.info("📈 SEÑAL DE COMPRA generada")
            return 'COMPRA'

        # Condiciones de VENTA
        elif (ultima['close'] < ultima['EMA20'] and 
              ultima['EMA20'] < ultima['EMA50'] and
              ultima['RSI14'] > 30 and
              ultima['Histogram'] < 0 and
              anterior['Histogram'] >= 0):
            logger.info("📉 SEÑAL DE VENTA generada")
            return 'VENTA'

        return 'NO TRADE'

    def ejecutar_compra(self, precio: float, timestamp: datetime):
        """Ejecuta una orden de compra"""
        if self.posicion_abierta:
            logger.warning("Ya hay una posición abierta")
            return

        if self.perdidas_consecutivas >= MAX_PERDIDAS_CONSECUTIVAS:
            logger.warning(f"Máximo de pérdidas consecutivas alcanzado ({MAX_PERDIDAS_CONSECUTIVAS})")
            return

        tamanio = self.calcular_tamanio_posicion(precio)
        costo = (precio * tamanio) * (1 + COMISION + SLIPPAGE)

        if costo > self.capital:
            logger.warning(f"Capital insuficiente. Requiere: ${costo:.2f}, Disponible: ${self.capital:.2f}")
            return

        self.posicion_abierta = True
        self.entrada_precio = precio
        self.entrada_tamanio = tamanio
        self.capital -= costo
        
        logger.info(f"✅ COMPRA EJECUTADA - Precio: ${precio:.2f}, Tamaño: {tamanio:.4f}, Capital: ${self.capital:.2f}")
        
        self.operaciones.append({
            'timestamp': timestamp,
            'tipo': 'COMPRA',
            'precio': precio,
            'tamanio': tamanio,
            'capital': self.capital
        })

    def ejecutar_venta(self, precio: float, timestamp: datetime) -> bool:
        """Ejecuta una orden de venta"""
        if not self.posicion_abierta:
            logger.warning("No hay posición abierta")
            return False

        ganancia_bruta = (precio - self.entrada_precio) * self.entrada_tamanio
        comisiones = (precio * self.entrada_tamanio) * (COMISION + SLIPPAGE)
        ganancia_neta = ganancia_bruta - comisiones
        
        self.capital += (self.entrada_precio * self.entrada_tamanio + ganancia_neta)
        retorno_porcentaje = (ganancia_neta / (self.entrada_precio * self.entrada_tamanio)) * 100

        if ganancia_neta < 0:
            self.perdidas_consecutivas += 1
            logger.warning(f"❌ PÉRDIDA - Retorno: {retorno_porcentaje:.2f}%")
        else:
            self.perdidas_consecutivas = 0
            logger.info(f"✅ GANANCIA - Retorno: {retorno_porcentaje:.2f}%")

        logger.info(f"🔴 VENTA EJECUTADA - Precio: ${precio:.2f}, Ganancia: ${ganancia_neta:.2f}, Capital: ${self.capital:.2f}")

        self.operaciones.append({
            'timestamp': timestamp,
            'tipo': 'VENTA',
            'precio': precio,
            'tamanio': self.entrada_tamanio,
            'ganancia': ganancia_neta,
            'retorno_porcentaje': retorno_porcentaje,
            'capital': self.capital
        })

        self.posicion_abierta = False
        self.entrada_precio = 0
        self.entrada_tamanio = 0
        self.historial_capital.append(self.capital)
        return True

    def aplicar_stop_loss(self, precio: float, timestamp: datetime, stop_loss_pct: float = 2):
        """Aplica stop loss automático"""
        if not self.posicion_abierta:
            return

        perdida_porcentaje = ((precio - self.entrada_precio) / self.entrada_precio) * 100
        
        if perdida_porcentaje < -stop_loss_pct:
            logger.info(f"🛑 STOP LOSS ACTIVADO - Pérdida: {perdida_porcentaje:.2f}%")
            self.ejecutar_venta(precio, timestamp)

    def aplicar_take_profit(self, precio: float, timestamp: datetime, take_profit_pct: float = 5):
        """Aplica take profit automático"""
        if not self.posicion_abierta:
            return

        ganancia_porcentaje = ((precio - self.entrada_precio) / self.entrada_precio) * 100
        
        if ganancia_porcentaje > take_profit_pct:
            logger.info(f"🎯 TAKE PROFIT ACTIVADO - Ganancia: {ganancia_porcentaje:.2f}%")
            self.ejecutar_venta(precio, timestamp)

    def procesar_candela(self, df: pd.DataFrame, precio_actual: float, timestamp: datetime):
        """Procesa una vela y ejecuta lógica de trading"""
        senal = self.analizar_mercado(df)

        if senal == 'COMPRA' and not self.posicion_abierta:
            self.ejecutar_compra(precio_actual, timestamp)
        elif senal == 'VENTA' and self.posicion_abierta:
            self.ejecutar_venta(precio_actual, timestamp)

        # Aplicar protecciones
        if self.posicion_abierta:
            self.aplicar_stop_loss(precio_actual, timestamp, stop_loss_pct=2)
            self.aplicar_take_profit(precio_actual, timestamp, take_profit_pct=5)

    def obtener_estadisticas(self) -> Dict:
        """Retorna estadísticas del bot"""
        operaciones_df = pd.DataFrame(self.operaciones)
        
        ventas = operaciones_df[operaciones_df['tipo'] == 'VENTA']
        ganancias_totales = ventas['ganancia'].sum() if len(ventas) > 0 else 0
        
        retorno_porcentaje = ((self.capital - self.capital_inicial) / self.capital_inicial) * 100
        
        return {
            'Capital Inicial': f"${self.capital_inicial:.2f}",
            'Capital Actual': f"${self.capital:.2f}",
            'Ganancias Totales': f"${ganancias_totales:.2f}",
            'Retorno %': f"{retorno_porcentaje:.2f}%",
            'Operaciones Totales': len(operaciones_df),
            'Operaciones Exitosas': len(ventas[ventas['ganancia'] > 0]) if len(ventas) > 0 else 0,
            'Operaciones Perdidas': len(ventas[ventas['ganancia'] < 0]) if len(ventas) > 0 else 0,
            'Posición Abierta': 'Sí' if self.posicion_abierta else 'No'
        }

    def generar_reporte(self):
        """Genera un reporte detallado"""
        stats = self.obtener_estadisticas()
        logger.info("\n" + "="*50)
        logger.info("REPORTE FINAL DEL BOT DE TRADING")
        logger.info("="*50)
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        logger.info("="*50 + "\n")


# =========================
# SIMULACIÓN DEL BOT
# =========================
def ejecutar_backtesting(candles: int = 100):
    """Ejecuta backtesting del bot"""
    logger.info(f"Iniciando backtesting con {candles} velas...")
    
    bot = TradingBot(CAPITAL_INICIAL)
    df = obtener_datos(candles=candles)

    for idx in range(30, len(df)):
        df_actual = df.iloc[:idx+1].copy()
        precio = df_actual.iloc[-1]['close']
        timestamp = df_actual.iloc[-1]['timestamp']

        bot.procesar_candela(df_actual, precio, timestamp)
        time.sleep(0.1)  # Pequeña pausa para simular tiempo real

    bot.generar_reporte()
    return bot


def ejecutar_trading_vivo(duracion_minutos: int = 10):
    """Ejecuta trading en tiempo real simulado"""
    logger.info(f"Iniciando trading en vivo por {duracion_minutos} minutos...")
    
    bot = TradingBot(CAPITAL_INICIAL)
    inicio = time.time()
    df = obtener_datos(candles=50)

    while (time.time() - inicio) < (duracion_minutos * 60):
        # Simular nuevas velas
        nuevo_precio = df.iloc[-1]['close'] * (1 + np.random.normal(0, 0.01))
        nueva_vela = {
            'close': nuevo_precio,
            'timestamp': datetime.now()
        }
        df = pd.concat([df, pd.DataFrame([nueva_vela])], ignore_index=True)

        bot.procesar_candela(df.iloc[-50:].reset_index(drop=True), nuevo_precio, datetime.now())
        
        time.sleep(TIEMPO_ENTRE_OPERACIONES)

    bot.generar_reporte()
    return bot


# =========================
# FUNCIÓN PRINCIPAL
# =========================
def main():
    logger.info("🤖 BOT DE TRADING AUTOMATIZADO INICIADO")
    logger.info(f"Capital Inicial: ${CAPITAL_INICIAL}")
    logger.info(f"Riesgo por operación: {RIESGO_POR_OPERACION*100}%")
    logger.info("="*50 + "\n")

    # Ejecutar backtesting
    bot = ejecutar_backtesting(candles=200)

    # Para trading en vivo descomenta:
    # bot = ejecutar_trading_vivo(duracion_minutos=10)


if __name__ == "__main__":
    main()
