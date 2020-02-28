from typing import List
from datetime import datetime


class Person(object):
    """
    Generic class to hold basic human information like name data
    """

    def __init__(self, first_name: str, last_name: str):
        self.first_name = first_name
        self.last_name = last_name
        self.name = first_name + " " + last_name

    def __str__(self):
        return "Name: {}\n".format(self.name)

    def __repr__(self):
        return self.__str__()


class Employee(Person):
    """
    Employee to attend to given flights
    """

    def __init__(self, first_name: str, last_name: str, position: str):
        super().__init__(first_name, last_name)
        self.position = position

    def __str__(self):
        str_rep = super().__str__()
        str_rep += "Position: {}".format(self.position)
        return str_rep

    def __repr__(self):
        return self.__str__()


class Passenger(Person):
    """
    Passenger to be aboard a given flight
    """

    def __init__(self, first_name: str, last_name: str, seat: str):
        super().__init__(first_name, last_name)
        self.seat = seat

    def __str__(self):
        str_rep = super().__str__()
        str_rep += "Seat: {}".format(self.seat)
        return str_rep

    def __repr__(self):
        return self.__str__()


class Flight():
    """
    Holds flight departure and arrival info
    Holds list of passengers aboard flight and employees manning it
    """

    GUID = 0  # Increments with each new instance of flight

    def __init__(self, departure_loc: str, departure_time: datetime, arrival_loc: str, arrival_time: datetime, existing_passengers: List[Passenger] = [], existing_employees: List[Employee] = []):
        self.departure_loc = departure_loc
        self.departure_time = departure_time
        self.arrival_loc = arrival_loc
        self.arrival_time = arrival_time

        self.passengers = existing_passengers
        self.employees = existing_employees

        self.id = Flight.get_GUID()
        Flight.increment_GUID()

    def __str__(self):
        str_rep = ""
        str_rep += "ID: {}\n".format(self.id)
        str_rep += "From {} at {}\nTo {} at {}\n".format(
            self.departure_loc, self.departure_time, self.arrival_loc, self.arrival_time)
        return str_rep

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.id == other.id

    def print_passenger_list(self) -> None:
        """Prints passenger list"""
        print("Passenger list for ", self)
        for passenger in self.passengers:
            print(passenger)

    def print_employee_list(self) -> None:
        """Prints employee list"""
        print("Employee list for ", self)
        for employee in self.employees:
            print(employee)

    def add_person(self, person) -> None:
        """Auto sorts type of person and handles storage accordingly"""
        if isinstance(person, Employee):
            self.employees.append(person)
        elif isinstance(person, Passenger):
            self.passengers.append(person)
        else:
            print("Could not determine role of ", person)

    @staticmethod
    def get_GUID() -> int:
        return Flight.GUID

    @classmethod
    def increment_GUID(cls) -> None:
        cls.GUID += 1


class AirlineBookingSystem():
    """Aggregator for instances of flights"""

    def __init__(self, existing_flights: List[Flight] = []):
        self.existing_flights = existing_flights

    def __str__(self):
        str_rep = ""
        str_rep += "Here are the flights for today:\n"
        for flight in self.existing_flights:
            str_rep += str(flight) + "\n"
        return str_rep

    def __repr__(self):
        str_rep = ""
        str_rep += "Here are the flights for today:\n"
        for flight in self.existing_flights:
            str_rep += str(flight) + "\n"
        return str_rep

    def add_flight(self, flight: Flight) -> None:
        """Adds given flight to flight list"""
        self.existing_flights.append(flight)
        print(flight, " has been added")

    def cancel_flight(self, flight: Flight) -> None:
        """Removes flight with same id as given flight"""
        self.existing_flights.remove(flight)
        print(flight, " has been removed\n")


if __name__ == "__main__":

    sys = AirlineBookingSystem()

    flight1 = Flight("Kansas City, MO", datetime(
        year=2020, month=2, day=22, hour=9), "Chicago, IL", datetime(
        year=2020, month=2, day=22, hour=10))
    flight1.add_person(Employee("Tina", "GoodLady", "Attendant"))
    flight1.add_person(Passenger("Landon", "Volkmann", "22A"))

    flight1.print_employee_list()
    flight1.print_passenger_list()

    sys.add_flight(flight1)

    flight2 = Flight("Kansas City, MO", datetime(
        year=2020, month=2, day=22, hour=9), "New York City, NY", datetime(
        year=2020, month=2, day=22, hour=12))
    flight2.add_person(Employee("Kevin", "Twinkle", "Attendant"))
    flight2.add_person(Employee("Sully", "Something", "Pilot"))
    flight2.add_person(Passenger("Trevor", "Klinkenberg", "10B"))
    flight2.add_person(Passenger("Brady", "Volkmann", "12C"))

    flight2.print_employee_list()
    flight2.print_passenger_list()

    sys.add_flight(flight2)

    print(sys)

    sys.cancel_flight(flight1)

    print(sys)
