from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, date

class OrderItemRow(BaseModel):
    orderItemId: str
    menuName: str
    price: int
    quantity: int

class OrderRow(BaseModel):
    orderId: str
    totalPrice: int
    orderChannel: str
    receiptMethod: str
    paymentMethod: str
    orderStatus: str
    requestMessage: Optional[str] = None
    createdAt: datetime
    storeName: str
    usersex: Optional[str] = None
    birthdate: Optional[date] = None
    isFirstOrderInStore: bool
    orderSeqInStore: int
    items: List[OrderItemRow]

class ReviewRow(BaseModel):
    rating: int
    content: str
    createdAt: datetime

class ReportGenerationRequest(BaseModel):
    storeId: str
    orders: List[OrderRow]
    reviews: List[ReviewRow]


class OrderPayload(BaseModel):
    storeId: str
    orders: List[OrderRow]
    
class ReviewPayload(BaseModel):
    storeId: str
    reviews: List[ReviewRow]

class GenerateJsonResponse(BaseModel):
    url: str
    localPath: Optional[str] = None
    createdAt: str
