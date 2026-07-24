from sqlalchemy import Column, Integer
from . import Base


class Item(Base):
    __tablename__ = "inferences"

    id = Column(Integer, primary_key=True, index=True)
