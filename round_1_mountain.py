import json
from datamodel import Listing, UserId, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
from copy import deepcopy
import collections
import statistics
import math

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
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
    VOLATILE_PRODUCTS = ["STARFRUIT", "ORCHIDS", "CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]
    PRODUCTS = STABLE_PRODUCTS + VOLATILE_PRODUCTS
    EMPTY_POSITION = {product: 0 for product in PRODUCTS}
    EMPTY_PRODUCT_LIST = {product: [] for product in PRODUCTS}
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    INF = 1e9
    STORAGE = 100

    old_vwaps = deepcopy(EMPTY_PRODUCT_LIST)

    @staticmethod
    def first(test_str):
        """Return the first element from an ordered collection
           or an arbitrary element from an unordered collection.
           Raise StopIteration if the collection is empty.

           Used to find the first element of an OrderedDict.
        """
        return next(iter(test_str))

    def compute_orders_stable(self, state: TradingState, product, order_depth):
        """For normal use, set ``selling_price`` and ``buying_price`` to 1e4"""
        logger.print(f"computing {product}...")
        orders: List[Order] = []
        position = state.position[product] if product in state.position.keys() else 0
        pos_limit = self.POSITION_LIMIT[product]

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        # unordered, uniterable sets of items
        bids = order_depth.buy_orders  # prices @ which bots want to sell
        asks = order_depth.sell_orders  # - buy

        # sorted dicts
        osell = collections.OrderedDict(sorted(asks.items()))
        obuy = collections.OrderedDict(sorted(bids.items(), reverse=True))

        # UNUSED
        ask_vol = sum(quote * qty for quote, qty in asks.items())
        bid_vol = sum(quote * qty for quote, qty in bids.items())
        ask_amt = sum(asks.values())
        bid_amt = sum(bids.values())
        vwaps = [ask_vol / ask_amt, bid_vol / bid_amt]
        vwap = statistics.mean(vwaps)

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        # not using it rn
        # update vwaps_over_time each time we run
        self.old_vwaps[product].append(vwap)
        if len(self.old_vwaps[product]) > self.STORAGE:
            self.old_vwaps[product].pop(0)
        # End of unused bits

        selling_price = int(1e4)
        buying_price = int(1e4)
        # selling_price = math.ceil(vwap)
        # buying_price = math.floor(vwap)

        # STRATEGY: copied from Stanford team

        best_ask = self.first(osell)
        # logger.print(best_ask)
        best_bid = self.first(obuy)

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

        t = 3/4 * pos_limit # hyperparameter telling us what to go till
        cpos = position  # variable used to ensure our orders are valid
        for ask, vol in osell.items():
            if ((ask < buying_price) or ((cpos < 0) and (ask == buying_price))) and cpos < pos_limit:
                order_for = min(-vol, pos_limit - cpos)
                cpos += order_for
                assert (order_for >= 0)
                logger.print("BUY", str(Order(product, ask, order_for)))
                orders.append(Order(product, ask, order_for))

        if cpos < pos_limit and position < 0:  # Occurs most frequently
            logger.print("FIRST BUY CONDITION")
            num = min(40, pos_limit - cpos)
            orders.append(Order(product, min(undercut_buy + 1, selling_price - 1), num))
            cpos += num
        elif cpos < pos_limit and position > t:  # Occurs least frequently
            logger.print("SECOND BUY CONDITION")
            num = min(40, pos_limit - cpos)
            orders.append(Order(product, min(undercut_buy - 1, selling_price - 1), num))
            cpos += num
        elif cpos < pos_limit:  # Intuitively this should be the "normal" case
            logger.print("THIRD BUY CONDITION")
            num = min(40, pos_limit - cpos)
            orders.append(Order(product, min(undercut_buy, selling_price - 1), num))
            cpos += num

        # Sell and ask
        cpos = position
        for bid, vol in obuy.items():
            if ((bid > selling_price) or ((position > 0) and (bid == selling_price))) and cpos > -pos_limit:
                order_for = min(vol, pos_limit + cpos)
                order_for *= -1  # we want to sell, so negate everything
                cpos += order_for
                assert (order_for <= 0)
                logger.print("SELL", str(Order(product, bid, order_for)))
                orders.append(Order(product, bid, order_for))

        if cpos > -pos_limit and position > 0:
            logger.print("FIRST SELL CONDITION")
            num = min(40, pos_limit + cpos)
            num *= -1
            orders.append(Order(product, max(undercut_sell - 1, buying_price + 1), num))
            cpos += num
        elif cpos > -pos_limit and position < -t:
            logger.print("SECOND SELL CONDITION")
            num = min(40, pos_limit + cpos)
            num *= -1
            orders.append(Order(product, max(undercut_sell + 1, buying_price + 1), num))
            cpos += num
        elif cpos > -pos_limit:
            logger.print("THIRD SELL CONDITION")
            num = min(40, pos_limit + cpos)
            num *= -1
            orders.append(Order(product, max(undercut_sell, buying_price + 1), num))
            cpos += num

        return orders

    def compute_orders_volatile(self, state, product, order_depth):
        """For normal use, set ``selling_price`` and ``buying_price`` to IDK... WHAT??"""
        logger.print("computing {product}...")
        orders: List[Order] = []
        position = state.position[product] if product in state.position.keys() else 0
        pos_limit = self.POSITION_LIMIT[product]

        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        # unordered, uniterable sets of items
        bids = order_depth.buy_orders  # prices @ which bots want to sell
        asks = order_depth.sell_orders  # - buy

        # sorted dicts
        osell = collections.OrderedDict(sorted(asks.items()))
        obuy = collections.OrderedDict(sorted(bids.items(), reverse=True))

        # STRATEGY: fair price is near the current vwap / past_vwap
        ask_vol = sum(quote * qty for quote, qty in asks.items())
        bid_vol = sum(quote * qty for quote, qty in bids.items())
        ask_amt = sum(asks.values())
        bid_amt = sum(bids.values())
        vwaps = [ask_vol / ask_amt, bid_vol / bid_amt]
        vwap = statistics.mean(vwaps)

        tariff = 0  # might as well have it here now
        epsilon = 0  # we think the maximum it can deviate that quickly

        past_vwap = sum(self.old_vwaps[product][-self.STORAGE:]) / self.STORAGE if len(self.old_vwaps) >= self.STORAGE else vwap
        assert past_vwap == vwap

        selling_price = min(vwap, past_vwap * (1 - epsilon)) + tariff / 2
        buying_price = max(vwap, past_vwap * (1 + epsilon)) - tariff / 2

        logger.print("Buying/Selling Prices: " + str([buying_price, selling_price]))
        logger.print("Buy Order depth: " + str(len(order_depth.buy_orders)) + ", Sell order depth: " + str(len(order_depth.sell_orders)))

        cpos = position  # variable used to ensure our orders are valid
        for ask, vol in osell.items():
            if ((ask < buying_price) or ((cpos < 0) and (ask == buying_price))) and cpos < \
                    self.POSITION_LIMIT[product]:
                order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                assert (order_for >= 0)
                logger.print("BUY", str(Order(product, ask, order_for)))
                orders.append(Order(product, ask, order_for))

        # TODO: implement fancy STARFRUIT buying strategy here

        cpos = position
        for bid, vol in obuy.items():
            if ((bid > selling_price) or ((position > 0) and (bid == selling_price))) and cpos > - \
                    self.POSITION_LIMIT[product]:
                order_for = min(vol, self.POSITION_LIMIT[product] + cpos)
                order_for *= -1  # we want to sell, so negate everything
                cpos += order_for
                assert (order_for <= 0)
                logger.print("SELL", str(Order(product, bid, order_for)))
                orders.append(Order(product, bid, order_for))

        # TODO: implement fancy STARFRUIT selling strategy here

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        # not using it rn
        # update vwaps_over_time each time we run
        self.old_vwaps[product].append(vwap)
        if len(self.old_vwaps[product]) > self.STORAGE:
            self.old_vwaps[product].pop(0)

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        if state.traderData != '':
            self.old_vwaps = json.loads(state.traderData)
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = deepcopy(self.EMPTY_PRODUCT_LIST)

        # UNUSED
        # Find acceptable prices for each product
        # WARNING: int(.) is needed to prevent typing errors
        amethysts_selling = int(1e4)
        amethysts_buying = int(1e4)

        starfruit_selling = int(-self.INF)
        starfruit_buying = int(self.INF)
        # END OF UNUSED

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            if product in self.STABLE_PRODUCTS:
                result[product] += self.compute_orders_stable(state, product, order_depth)
            elif product in self.VOLATILE_PRODUCTS:
                result[product] += self.compute_orders_volatile(state, product, order_depth)
            else:
                raise ValueError(f"{product} is an invalid product!")

        trader_data = json.dumps(self.old_vwaps)

        conversions = 0  # UNUSED

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data
