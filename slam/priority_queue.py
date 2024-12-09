import heapq


class PriorityQueue:
    def __init__(self):
        """Initialize an empty priority queue."""
        self.elements = []

    def empty(self):
        """
        Check if the priority queue is empty.
        :return: True if the queue is empty, False otherwise.
        """
        return not self.elements

    def put(self, item, priority):
        """
        Add an item to the queue with a given priority.
        :param item: The item to be added.
        :param priority: The priority of the item (lower values indicate higher priority).
        """
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        """
        Remove and return the item with the highest priority.
        :return: The item with the highest priority.
        """
        return heapq.heappop(self.elements)[1]
