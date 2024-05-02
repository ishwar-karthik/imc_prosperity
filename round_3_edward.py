import json
from datamodel import Listing, UserId, Observation, Order, OrderDepth, \
    ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from typing import List, Any
from copy import deepcopy
import collections
import statistics
import math
import numpy as np

import timeit


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self,
              state: TradingState,
              orders: dict[Symbol,
                           list[Order]],
              conversions: int,
              trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same
        # max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(
                state, self.truncate(
                    state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(
            self,
            state: TradingState,
            trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self,
                          listings: dict[Symbol,
                                         Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self,
                              order_depths: dict[Symbol,
                                                 OrderDepth]) -> dict[Symbol,
                                                                      list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders,
                order_depth.sell_orders]

        return compressed

    def compress_trades(self,
                        trades: dict[Symbol,
                                     list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self,
                        orders: dict[Symbol,
                                     list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()


class Trader:
    STABLE_PRODUCTS = ["AMETHYSTS"]
    STABLE_SELLING_PRICES = {product: int(1e4) for product in STABLE_PRODUCTS}
    STABLE_BUYING_PRICES = STABLE_SELLING_PRICES
    VOLATILE_PRODUCTS = [
        "STARFRUIT",
        "ORCHIDS",
        "CHOCOLATE",
        "STRAWBERRIES",
        "ROSES"]
    MOMENTUM_PRODUCTS = ["GIFT_BASKET"]
    PRODUCTS = STABLE_PRODUCTS + VOLATILE_PRODUCTS + MOMENTUM_PRODUCTS
    EMPTY_POSITION = {product: 0 for product in PRODUCTS}
    EMPTY_PRODUCT_LIST = {product: [] for product in PRODUCTS}
    POSITION_LIMITS = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
        'CHOCOLATE': 250,
        'STRAWBERRIES': 350,
        'ROSES': 60,
        'GIFT_BASKET': 60,
        'ORCHIDS': 100}
    INF = 1e9
    STORAGE = 100
    STARTING_STORAGE = 300
    KEEP_STARTING_VWAPS = {product: False for product in PRODUCTS}
    for product in VOLATILE_PRODUCTS:
        KEEP_STARTING_VWAPS[product] = True
    PERCENTILE_LOWER = 30
    PERCENTILE_UPPER = 70

    old_vwaps = deepcopy(EMPTY_PRODUCT_LIST)
    starting_vwaps = deepcopy(EMPTY_PRODUCT_LIST)
    old_positions = deepcopy(EMPTY_PRODUCT_LIST)
    avg_positions = deepcopy(EMPTY_POSITION)

    percentile_lower = -INF
    percentile_upper = INF

    exponential_ma_12 = 0
    exponential_ma_26 = 0
    macd = 0

    # shallow copy so that we never update this manually
    trader_data_dict = {
        "old_vwaps": old_vwaps,
        "starting_vwaps": starting_vwaps,
        "old_positions": old_positions,
        "avg_positions": avg_positions,
        "percentile_lower": percentile_lower,
        "percentile_upper": percentile_upper,
        "exponential_ma_12": exponential_ma_12,
        "exponential_ma_26": exponential_ma_26,
        "macd": macd,
    }

    @staticmethod
    def first(test_str):
        """Return the first element from an ordered collection
           or an arbitrary element from an unordered collection.
           Raise StopIteration if the collection is empty.

           Used to find the first element of an OrderedDict.
        """
        return next(iter(test_str))

    """ UNUSED
    def time_this_method(self, function):
        def wrapper(*args, **kwargs):
            start = timeit.default_timer()
            original_return_val = function(*args, **kwargs)
            end = timeit.default_timer()
            self.trader_data_dict["timer"].append(end - start)
            return original_return_val
        return wrapper
    """

    def update_trader_data(self, state, product, data):
        if "vwap" in data.keys():
            vwap = data["vwap"]
            self.old_vwaps[product].append(vwap)
            if len(self.old_vwaps[product]) > self.STORAGE:
                self.old_vwaps[product].pop(0)
            if self.KEEP_STARTING_VWAPS[product]:
                if state.timestamp <= self.STARTING_STORAGE * 100:
                    self.starting_vwaps[product].append(vwap)
                    if state.timestamp == self.STARTING_STORAGE * 100:
                        # raise Exception(f"reached at timestamp={state.timestamp}")
                        self.trader_data_dict["percentile_lower"] = np.percentile(self.starting_vwaps[product], self.PERCENTILE_LOWER)
                        self.trader_data_dict["percentile_upper"] = np.percentile(self.starting_vwaps[product], self.PERCENTILE_UPPER)
                        del self.starting_vwaps  # conserve memory
        if "position" in data.keys():
            pos = data["position"]
            self.old_positions[product].append(pos)
            if len(self.old_positions[product]) > self.STORAGE:
                self.old_positions[product].pop(0)
                self.avg_positions = statistics.mean(self.old_positions[product])

    def compute_orders_stable(self, state: TradingState, product, order_depth):
        """For normal use, set ``selling_price`` and ``buying_price`` to 1e4"""
        logger.print(f"computing {product}...")
        orders: List[Order] = []
        position = state.position[product] if product in state.position.keys(
        ) else 0
        pos_limit = self.POSITION_LIMITS[product]

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        # unordered, uniterable sets of items
        bids = order_depth.buy_orders  # prices @ which bots want to sell
        asks = order_depth.sell_orders  # - buy

        # sorted dicts
        osell = collections.OrderedDict(sorted(asks.items()))
        obuy = collections.OrderedDict(sorted(bids.items(), reverse=True))

        # get the prices from our global vars
        selling_price = self.STABLE_SELLING_PRICES[product]
        buying_price = self.STABLE_BUYING_PRICES[product]

        best_bid = self.first(obuy)
        best_ask = self.first(osell)

        # shift by one to ensure a profit
        undercut_sell = best_ask - 1
        undercut_buy = best_bid + 1

        # Buy and bid
        """First we trade all the acceptable ASK orders from the order book.
        If we still haven't reached the position limit after that, we'll put additional (undercut) orders on the market
        in either of 3 cases: (for now assume we're buying)
        Define `undercut_buy = best_bid+1`. We want to put orders at `undercut_buy` to beat the competition, but this is
        balanced by the fact that we only buy if we can sell it later with profit. Hence we will buy at around
        `min(undercut_buy, selling_price-1)`.
        After some fine-tuning:
        (1) If we have cpos < 0 (able to buy a lot), buy at `min(undercut_buy+1, selling_price-1)`
        (2) If cpos > 15 (only a small quota to buy, close to upper position limit), buy at `min(undercut_buy-1, selling_price-1)`
        (3) If 0 <= cpos <= 15, buy at `min(undercut_buy, selling_price-1)`
        """

        t = (3/4) * pos_limit  # hyperparameter telling us what to go till
        cpos = position  # variable used to ensure our orders are valid
        for ask, vol in osell.items():
            if ((ask < buying_price) or ((cpos < 0) and (
                    ask == buying_price))) and cpos < pos_limit:
                order_for = min(-vol, pos_limit - cpos)
                cpos += order_for
                assert (order_for >= 0)
                logger.print("BUY", str(Order(product, ask, order_for)))
                orders.append(Order(product, ask, order_for))

        if cpos < pos_limit and position < 0:  # Occurs most frequently
            logger.print("FIRST BUY CONDITION")
            num = min(2 * pos_limit, pos_limit - cpos)
            orders.append(
                Order(
                    product, min(
                        undercut_buy + 1, selling_price - 1), num))
            cpos += num
        elif cpos < pos_limit and position > t:  # Occurs least frequently
            logger.print("SECOND BUY CONDITION")
            num = min(2 * pos_limit, pos_limit - cpos)
            orders.append(
                Order(
                    product, min(
                        undercut_buy - 1, selling_price - 1), num))
            cpos += num
        elif cpos < pos_limit:  # Intuitively this should be the "normal" case
            logger.print("THIRD BUY CONDITION")
            num = min(2 * pos_limit, pos_limit - cpos)
            orders.append(
                Order(
                    product, min(
                        undercut_buy, selling_price - 1), num))
            cpos += num

        # Sell and ask
        cpos = position
        for bid, vol in obuy.items():
            if ((bid > selling_price) or ((position > 0) and (
                    bid == selling_price))) and cpos > -pos_limit:
                order_for = min(vol, pos_limit + cpos)
                order_for *= -1  # we want to sell, so negate everything
                cpos += order_for
                assert (order_for <= 0)
                logger.print("SELL", str(Order(product, bid, order_for)))
                orders.append(Order(product, bid, order_for))

        if cpos > -pos_limit and position > 0:
            logger.print("FIRST SELL CONDITION")
            num = min(2 * pos_limit, pos_limit + cpos)
            num *= -1
            orders.append(
                Order(
                    product, max(
                        undercut_sell - 1, buying_price + 1), num))
            cpos += num
        elif cpos > -pos_limit and position < -t:
            logger.print("SECOND SELL CONDITION")
            num = min(2 * pos_limit, pos_limit + cpos)
            num *= -1
            orders.append(
                Order(
                    product, max(
                        undercut_sell + 1, buying_price + 1), num))
            cpos += num
        elif cpos > -pos_limit:
            logger.print("THIRD SELL CONDITION")
            num = min(2 * pos_limit, pos_limit + cpos)
            num *= -1
            orders.append(
                Order(
                    product, max(
                        undercut_sell, buying_price + 1), num))
            cpos += num

        return orders

    def orchid_production(self, obs: ConversionObservation):
        '''Computes orchid productivity given a conversion observation.'''
        sun_def = 7 - (obs.sunlight * (12/2500)) # how many hours below 7 does the orchid lack
        sun_factor = 1 if sun_def <= 0 else math.pow(0.96, sun_def * 6)

        humidity_deviation = abs(obs.humidity - 70) - 10 # deviation of humidity from 60-80% range
        humidity_factor = 1 if humidity_deviation <= 0 else math.pow(0.98, humidity_deviation/5)

        return sun_factor * humidity_factor

    def compute_orders_volatile(self, state: TradingState, product, order_depth):
        """For normal use, set ``selling_price`` and ``buying_price`` to IDK... WHAT??"""
        logger.print(f"computing {product}...")
        orders: List[Order] = []
        position = state.position[product] if product in state.position.keys() else 0
        pos_limit = self.POSITION_LIMITS[product]

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        # unordered, uniterable sets of items
        bids = order_depth.buy_orders  # prices @ which bots want to sell
        asks = order_depth.sell_orders  # - buy

        # sorted dicts
        osell = collections.OrderedDict(sorted(asks.items()))
        obuy = collections.OrderedDict(sorted(bids.items(), reverse=True))

        # STRATEGY: fair price is near the current vwap / past_vwap
        best_bid = self.first(obuy)
        best_ask = self.first(osell)
        ask_vol = sum(quote * qty for quote, qty in asks.items())
        bid_vol = sum(quote * qty for quote, qty in bids.items())
        ask_amt = sum(asks.values())
        bid_amt = sum(bids.values())
        vwaps = [ask_vol / ask_amt, bid_vol / bid_amt]
        vwap = statistics.mean(vwaps)

        all_obs = state.observations.conversionObservations
        obs = all_obs[product] if product in all_obs else ConversionObservation(0, 0, 0, 0, 0, 0, 0)
        tariff = obs.importTariff + obs.exportTariff + obs.transportFees  # might as well have it here now
        epsilon = 0  # we think the maximum it can deviate that quickly

        '''
        if product == "ORCHIDS":
            production_weight = 40
            mean_prod = 0.9
            vwap -= (self.orchid_production(obs) - mean_prod)*production_weight
        '''

        past_vwap = sum(self.old_vwaps[product][-self.STORAGE:]) / \
            self.STORAGE if len(self.old_vwaps) >= self.STORAGE else vwap
        assert past_vwap == vwap  # this line tests when

        selling_price = min(vwap, past_vwap * (1 - epsilon)) + tariff / 2
        buying_price = max(vwap, past_vwap * (1 + epsilon)) - tariff / 2

        logger.print("Buying/Selling Prices: " +
                     str([buying_price, selling_price]))
        logger.print("Buy Order depth: " +
                     str(len(order_depth.buy_orders)) +
                     ", Sell order depth: " +
                     str(len(order_depth.sell_orders)))

        """STRATEGY: trade on historical percentiles"""
        best_bid = self.first(obuy)
        best_ask = self.first(osell)

        cpos_buying = position  # variable used to ensure our orders are valid
        for ask, vol in osell.items():
            if (ask < buying_price) or (
                    (cpos_buying < 0) and (ask == buying_price)):
                if cpos_buying < pos_limit:
                    order_for = min(-vol, pos_limit - cpos_buying)
                    cpos_buying += order_for
                    assert (order_for >= 0)
                    logger.print("BUY", str(Order(product, ask, order_for)))
                    orders.append(Order(product, ask, order_for))

        # shift by one to ensure a profit
        undercut_buy = best_bid + 1

        cpos_selling = position
        for bid, vol in obuy.items():
            if (bid > selling_price) or (
                    (cpos_selling > 0) and (bid == selling_price)):
                if cpos_selling > -pos_limit:
                    order_for = min(vol, pos_limit + cpos_selling)
                    order_for *= -1  # we want to sell, so negate everything
                    cpos_selling += order_for
                    assert (order_for <= 0)
                    logger.print("SELL", str(Order(product, bid, order_for)))
                    orders.append(Order(product, bid, order_for))

        # shift by one to ensure a profit
        undercut_sell = best_ask - 1

        # Compute historical percentiles
        if state.timestamp > self.STARTING_STORAGE * 100:
            oversold = self.trader_data_dict["percentile_lower"]
            overbought = self.trader_data_dict["percentile_upper"]

            # if cpos_buying < pos_limit and rsi < oversold_rsi:
            if cpos_buying < pos_limit and vwap < oversold:
                logger.print("OVERSOLD")
                buy_momentum = min(undercut_buy, math.ceil(selling_price) - 1)
                assert isinstance(buy_momentum, int)
                order_for = pos_limit - cpos_buying
                cpos_buying += order_for
                assert (order_for >= 0)
                logger.print("BUY", str(Order(product, buy_momentum, order_for)))
                orders.append(Order(product, buy_momentum, order_for))

            # if cpos_selling > -pos_limit and rsi > overbought_rsi:
            if cpos_buying < pos_limit and vwap > overbought:
                logger.print("OVERBOUGHT")
                sell_momentum = max(undercut_sell, math.floor(buying_price) + 1)
                assert isinstance(sell_momentum, int)
                order_for = pos_limit + cpos_selling
                order_for *= -1
                cpos_selling += order_for
                assert (order_for <= 0)
                logger.print("SELL", str(Order(product, sell_momentum, order_for)))
                orders.append(Order(product, sell_momentum, order_for))

        # update old_vwaps each time we run
        self.update_trader_data(state, product, {"vwap": vwap, "position": position})

        return orders

    @staticmethod
    def exponential_moving_average(prev_ema, new_value, period):
        exp_weight = 2/(1 + period)
        return exp_weight * new_value + (1 - exp_weight) * prev_ema

    def compute_orders_momentum(
            self,
            state: TradingState,
            product,
            order_depth):
        """For normal use, set ``selling_price`` and ``buying_price`` to...
        there is no normal use. You're on your own, buddy. """
        logger.print(f"computing {product}...")
        orders: List[Order] = []
        position = state.position[product] if product in state.position.keys(
        ) else 0
        pos_limit = self.POSITION_LIMITS[product]

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        # unordered, uniterable sets of items
        bids = order_depth.buy_orders  # prices @ which bots want to sell
        asks = order_depth.sell_orders  # - buy

        # sorted dicts
        osell = collections.OrderedDict(sorted(asks.items()))
        obuy = collections.OrderedDict(sorted(bids.items(), reverse=True))

        # STRATEGY: fair price is near the current vwap / past vwap
        best_bid = self.first(obuy)
        best_ask = self.first(osell)
        ask_vol = sum(quote * qty for quote, qty in asks.items())
        bid_vol = sum(quote * qty for quote, qty in bids.items())
        ask_amt = sum(asks.values())
        bid_amt = sum(bids.values())
        vwaps = [ask_vol / ask_amt, bid_vol / bid_amt]
        vwap = statistics.mean(vwaps)

        all_obs = state.observations.conversionObservations
        obs = all_obs[product] if product in all_obs else ConversionObservation(0, 0, 0, 0, 0, 0, 0)

        tariff = obs.importTariff + obs.exportTariff + obs.transportFees
        epsilon = 0  # we think the maximum it can deviate in the long-term

        space = self.STORAGE
        past_vwap = sum(self.old_vwaps[product][-space:]) / \
            space if len(self.old_vwaps) >= space else vwap

        logger.print("old_vwaps:", self.old_vwaps[product])
        selling_price = min(vwap, past_vwap * (1 - epsilon)) + tariff / 2
        buying_price = max(vwap, past_vwap * (1 + epsilon)) - tariff / 2

        logger.print("Buying/Selling Prices: " +
                     str([buying_price, selling_price]))
        logger.print("Buy Order depth: " +
                     str(len(order_depth.buy_orders)) +
                     ", Sell order depth: " +
                     str(len(order_depth.sell_orders)))

        # Compute RSI

        differences = np.diff(self.old_vwaps[product][-self.STORAGE:])
        # logger.print("differences:", differences)
        gains = differences[differences > 0]
        # logger.print("gains:", gains)
        losses = -differences[differences < 0]
        average_gain = np.mean(gains) if len(gains) > 0 else 0
        average_loss = np.mean(losses) if len(losses) > 0 else 1
        rs = average_gain / average_loss
        rsi = 100 - 100 / (1 + (rs))  # rsi calculation
        logger.print("RSI:", rsi)

        cpos_buying = position  # variable used to ensure our orders are valid
        for ask, vol in osell.items():
            if (ask < buying_price) or (
                    (cpos_buying < 0) and (ask == buying_price)):
                if cpos_buying < pos_limit:
                    order_for = min(-vol, pos_limit - cpos_buying)
                    cpos_buying += order_for
                    assert (order_for >= 0)
                    logger.print("BUY", str(Order(product, ask, order_for)))
                    orders.append(Order(product, ask, order_for))

        # shift by one to ensure a profit
        undercut_buy = best_bid + 1

        cpos_selling = position
        for bid, vol in obuy.items():
            if (bid > selling_price) or (
                    (cpos_selling > 0) and (bid == selling_price)):
                if cpos_selling > -pos_limit:
                    order_for = min(vol, pos_limit + cpos_selling)
                    order_for *= -1  # we want to sell, so negate everything
                    cpos_selling += order_for
                    assert (order_for <= 0)
                    logger.print("SELL", str(Order(product, bid, order_for)))
                    orders.append(Order(product, bid, order_for))

        # shift by one to ensure a profit
        undercut_sell = best_ask - 1

        """STRATEGY: trade on MACD"""
        if state.timestamp == 1200:
            simple_ma = statistics.mean(self.old_vwaps[product])
            self.exponential_ma_12 = simple_ma
            self.trader_data_dict["exponential_ma_12"] = self.exponential_ma_12
        elif state.timestamp > 1200:
            self.exponential_ma_12 = self.exponential_moving_average(self.exponential_ma_12, vwap, 12)

        if state.timestamp == 2600:
            simple_ma = statistics.mean(self.old_vwaps[product])
            self.exponential_ma_26 = simple_ma
            self.trader_data_dict["exponential_ma_26"] = self.exponential_ma_26
        elif state.timestamp > 2600:
            self.exponential_ma_12 = self.exponential_moving_average(self.exponential_ma_12, vwap, 12)

        if state.timestamp > 2600:
            self.macd = self.exponential_ma_12 - self.exponential_ma_26
            self.trader_data_dict["macd"] = self.macd

        self.update_trader_data(state, product, {"vwap": vwap})

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        if state.traderData != '':
            self.trader_data_dict = json.loads(state.traderData)
            self.old_vwaps = self.trader_data_dict["old_vwaps"]
            self.starting_vwaps = self.trader_data_dict["starting_vwaps"]
            self.old_positions = self.trader_data_dict["old_positions"]
            self.percentile_lower = self.trader_data_dict["percentile_lower"]
            self.percentile_upper = self.trader_data_dict["percentile_upper"]
            # raise Exception(f"reached at time={state.timestamp}")
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = deepcopy(self.EMPTY_PRODUCT_LIST)

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            if product in self.STABLE_PRODUCTS:
                pass  # result[product] += self.compute_orders_stable(state, product, order_depth)
            elif product in self.VOLATILE_PRODUCTS:
                result[product] += self.compute_orders_volatile(
                    state, product, order_depth)
            elif product in self.MOMENTUM_PRODUCTS:
                result[product] += self.compute_orders_momentum(
                    state, product, order_depth)
            else:
                raise ValueError(f"{product} is an invalid product!")

        trader_data = json.dumps(self.trader_data_dict)

        conversions = 0  # UNUSED

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data
