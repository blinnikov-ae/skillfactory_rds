/*Подзапрос релевантой информации об аэропортах*/
with arpts as
(
select
    ap.airport_code,
    ap.city,
    ap.longitude,
    ap.latitude
from dst_project.airports ap
),
/*Подзапрос основной информации о самолетах*/
adj_aircrafts as
(
select
    ac.aircraft_code,
    ac.model,
    ac.range,
    count(s.seat_no) total_seats,
    (
	select 
		count(s.seat_no)
	from 
		dst_project.seats s 
	where 
		ac.aircraft_code = s.aircraft_code
		and s.fare_conditions = 'Business'
	) b_seats,
	(
	select 
		count(s.seat_no)
	from 
		dst_project.seats s 
	where 
		ac.aircraft_code = s.aircraft_code
		and s.fare_conditions = 'Comfort'
	) c_seats,
	(
	select 
		count(s.seat_no)
	from 
		dst_project.seats s 
	where 
		ac.aircraft_code = s.aircraft_code
		and s.fare_conditions = 'Economy'
	) e_seats
from dst_project.aircrafts ac
    join dst_project.seats s
        on ac.aircraft_code = s.aircraft_code
group by 1,2,3
),
/*Подзапрос аагрегированной информации по рейсам из ticket_flights*/
tckts as
(
select
    tf.flight_id,
    count(tf.ticket_no)total_tickets,
    sum(tf.amount) revenues_per_flight,
    count(case when tf.fare_conditions = 'Business' then tf.fare_conditions end) b_tickets,
    count(case when tf.fare_conditions = 'Comfort' then tf.fare_conditions end) c_tickets,
    count(case when tf.fare_conditions = 'Economy' then tf.fare_conditions end) e_tickets
from dst_project.ticket_flights tf
group by 1
)
/*Вывод итоговых данных*/
select
   f.flight_id,
   f.flight_no,
   f.scheduled_departure,
   f.scheduled_arrival,
   extract(hour from (f.actual_arrival - f.actual_departure))*60
    + extract(minute from (f.actual_arrival - f.actual_departure)) actual_flight_duration,
   f.departure_airport,
   ap_d.city departure_city,
   ap_d.longitude depature_longitude,
   ap_d.latitude departure_latitude,
   f.arrival_airport,
   ap_a.city arrival_city,
   ap_a.longitude arrival_longitude,
   ap_a.latitude arrival_latitude,
   aa.aircraft_code,
   aa.model,
   aa.range,
   aa.total_seats,
   aa.b_seats,
   aa.c_seats,
   aa.e_seats,
   t.b_tickets,
   t.c_tickets,
   t.e_tickets,
   t.total_tickets,
   t.revenues_per_flight
from dst_project.flights f
    join arpts ap_d
        on f.departure_airport = ap_d.airport_code
    join arpts ap_a
        on f.arrival_airport = ap_a.airport_code
    join adj_aircrafts aa
        on f.aircraft_code = aa.aircraft_code
    join tckts t
        on f.flight_id = t.flight_id
where ((extract(month from f.scheduled_departure) > 11 
        or extract(month from f.scheduled_departure) < 2)
        or (extract(month from f.scheduled_arrival) > 11 
        or extract(month from f.scheduled_arrival) < 2))
    and (f.departure_airport = 'AAQ' or f.arrival_airport = 'AAQ')
    and f.status not in ('Cancelled')