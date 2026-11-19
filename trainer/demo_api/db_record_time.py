"""Existing record time fields, in preference order; absent times stay unknown."""
TIME_COLUMNS = ('created_at', 'received_at', 'requested_at', 'imported_at', 'submitted_at', 'started_at')
MISSING_TIME = '__created_at'


def first_record_time(columns):
    return next((name for name in TIME_COLUMNS if name in columns), MISSING_TIME)
