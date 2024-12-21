from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Plate(Base):
    __tablename__ = "plates"
    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, index=True)
