select
    t.book_ref,
    avg(b.total_amount)/sum(tf.amount)
from dst_project.tickets t
    join dst_project.bookings b
        on t.book_ref = b.book_ref
    join dst_project.ticket_flights tf
        on t.ticket_no = tf.ticket_no
group by 1
    having avg(b.total_amount)/sum(tf.amount) != 1