# Where Am I? Location and Positioning

Devices with GPS chips are everywhere. You can even track your cat or chicken! In this chapter, you will learn how to use Qt for location and positioning services.

Qt Positioning entails geographic coordinates from various sources, including satellites, Wi-Fi, and log files. Qt Location is all about local places, for example services, such as restaurants or public parks, and also routing information.

In this chapter, we will cover the following topics:

*   Positioning with satellites
*   Mapping the positions
*   Places of interest

# Positioning with satellites

A phone usually has a built-in GPS modem but also other sources of positioning information, so I will use Android for this example. The main Qt classes we will look at are as follows:

Here are the Qt Positioning APIs:

*   `QGeoSatelliteInfo`
*   `QGeoLocation`
*   `QGeoPositionInfoSource`
*   `QGeoCoordinate`

And here are the Qt Location APIs:

*   `QPlaceSearchResult`
*   `QPlaceContent`
*   `QGeoRoute`

First, we need to edit the `.pro` file and add `QT += positioning`.

# QGeoSatelliteInfoSource

You can show the user satellite information by using `QGeoSatelliteInfoSource`, which has a `static` method to get `QGeoSatelliteInfoSource.`

The source code can be found on the Git repository under the `Chapter08-1` directory, in the `cp8` branch.

We will start by calling `QGeoSatelliteInfoSource::createDefaultSource`.

```cpp
QGeoSatelliteInfoSource *source = QGeoSatelliteInfoSource::createDefaultSource(this);
```

On some systems, such as iOS, satellite information is not exposed to the public API, so `QGeoSatelliteInfoSource` will not work on that platform.

This constructs a `QGeoSatelliteInfoSource` object for the highest-priority plugin on the system, which is about the same as doing the following:

```cpp
QStringList geoSources = QGeoSatelliteInfoSource::availableSources();
QGeoSatelliteInfoSource *source = QGeoSatelliteInfoSource::createSource(geoSources.at(0),this);
```

There are two signals of particular interest: `satellitesInUseUpdated` and `satellitesInViewUpdated`. In addition, there is the overloaded `error` signal, so we need to use the special `QOverload` syntax:

```cpp
connect(source, QOverload<QGeoSatelliteInfoSource::Error>::
    of(&QGeoSatelliteInfoSource::error), 
    this, &SomeClass::error);
```

The `satellitesInUseUpdated` signal is emitted when the number of satellites that the system is using changes. The `satellitesInViewUpdated` signal gets emitted when the number of satellites the system can see changes. We will receive a list of `QGeoSatelliteInfo` objects.

# QGeoSatelliteInfo

Let's connect the `satellitesInViewUpdated` signal so we can detect when satellites are found:

```cpp
connect(source, SIGNAL(satellitesInViewUpdated(QList<QGeoSatelliteInfo>)),
    this, SLOT(satellitesInViewUpdated(QList<QGeoSatelliteInfo>)));
```

We can receive information for individual satellites this way. Information such as a satellite identifier, signal strength, elevation, and azimuth is included:

```cpp
void SomeClass::satellitesInViewUpdated(const QList<QGeoSatelliteInfo> &infos)
{
    if (infos.count() > 0)
        qWarning() << "Number of satellites in view:" << infos.count();

    foreach (const QGeoSatelliteInfo &info, infos) {
        qWarning() << "    "
            << "satelliteIdentifier" << info.satelliteIdentifier()
            << "signalStrength" << info.signalStrength()
            << (info.hasAttribute(QGeoSatelliteInfo::Elevation) ?  "Elevation "
+ QString::number(info.attribute(QGeoSatelliteInfo::Elevation)) : "")
            << (info.hasAttribute(QGeoSatelliteInfo::Elevation) ?  "Azimuth " +
QString::number(info.attribute(QGeoSatelliteInfo::Azimuth)) : "");
    }
}
```

There's a lot to see here on a small screen. Every update is a new line, and you can see as it locates and adds different satellites when they come into view:

![](img/e67199f7-ab26-4d0e-84eb-0aa5d209cc9e.png)

The next step is to use those satellites to triangulate our position on the globe. We start by using `QGeoPositionInfoSource`.

# QGeoPositionInfoSource

We can get the latitude and longitude position of the device by using `QGeoPositionInfoSource`, which encapsulates positional data. Like `QGeoSatelliteInfoSource`, it has two `static` methods to create the `source` object:

The source code can be found on the Git repository under the `Chapter08-2` directory, in the `cp8` branch.

```cpp
QGeoPositionInfoSource *geoSource = QGeoPositionInfoSource::createDefaultSource(this);
```

The `QGeoPositionInfoSource` signal we are interested in is `positionUpdated(const QGeoPositionInfo &update)`:

```cpp
connect(geoSource, &QGeoPositionInfoSource::positionUpdated,
    this, &MainWindow::positionUpdated);
```

To start receiving updates, call `startUpdates();`:

```cpp
 geoSource->startUpdates();
```

The `positionUpdated` signal receives a `QGeoPositionInfo`.

# QGeoPositionInfo

`QGeoPositionInfo` contains a `QGeoCoordinate` that contains our latitude and longitude coordinates, as well as a timestamp for the location data.

It can also contain the following optional attributes:

*   `Direction`
*   `GroundSpeed`
*   `VerticalSpeed`
*   `MagneticVariation`

*   `HorizontalAccuracy`
*   `VerticalAccuracy`

The attributes can be checked with `hasAttribute(QGeoPositionInfo::Attribute)` and retrieved with the `attribute(QGeoPositionInfo::Attribute)` function:

```cpp
if (positionInfo.hasAttribute(QGeoPositionInfo::MagneticVariation)
    qreal magneticVariation = positionInfo.attribute(QGeoPositionInfo::MagneticVariation);
```

To get latitude and longitude information, call the `coordinate()` function in the `QGeoPositionInfo` class, which returns a `QGeoCoordinate`.

# QGeoCoordinate

`QGeoCoordinate` contains the latitude and longitude coordinates, and can be found calling the respective `latitude()` and `longitude()` functions. It can be made up of different types of data, and can be discovered by calling the `type()` function, which returns an `enum` of `QGeoCoordinate::CoordinateType`, which can be one of the following values:

*   `InvalidCoordinate`: Invalid coordinate
*   `Coordinate2D`: Contains latitude and longitude coordinates

*   `Coordinate3D`: Contains latitude, longitude, and altitude coordinates

We can get the `QGeoCoordinate` from the `QGeoPositionInfo` object's `coordinate()` function which, in turn, has `latitude` and `longitude` values:

```cpp
QGeoCoordinate coords = positionInfo.coordinate();
QString("Latitude %1\n").arg(coords.latitude());
QString("Longitude %1\n").arg(coords.longitude()); 

​if (coords.type() == QGeoCoordinate::Coordinate3D)
    QString("Altitude %1\n").arg(coords.altitude())
```

Let's take a look at how we do this using Qt Quick and QML.

# Qt Quick

There are corresponding QML elements available for positioning.

The `import` statement would be `import QtPositioning 5.12`.

Let's do the same simple thing with QML and show our latitude and longitude values.

Here are the Qt Quick item equivalents of the previously-mentioned classes:

*   `PositionSource`: `QGeoPositionInfoSource`

*   `Position`: `QGeoPositionInfo`

*   `Coordinate`: `QGeoCoordinate`

Qt Quick is often much simpler, and quick to implement these things.

The source code can be found on the Git repository under the `Chapter08-3` directory, in the `cp8` branch.

We implement `PositionSource` with an `updateInterval` of 1,000, which means the devices position will update every 1,000 milliseconds. We set it to `active` to start the updates:

```cpp
    PositionSource {
        id: positionSource
        updateInterval: 1000
        active: true
```

This component has a signal named `onPositionChanged`, which gets called when the position changes. We receive the changed coordinates and can then use them:

```cpp
        onPositionChanged: {
            var coord = positionSource.position.coordinate;
            console.log("Coordinate:", coord.longitude, coord.latitude);
            latitudeLabel.text = "Latitude: " + coord.latitude;
            longitudeLabel.text = "Longitude: " + coord.longitude;
            if (positionSource.position.altitudeValid)
                altitudeLabel.text = "Altitude: " + coord.altitude;
      }
 }
```

Now that we know where we are, we can use those location details to get certain details around the coordinates, like the map and location's place details.

# Mapping the positions

We now need a map of some sort to show our location findings.

The `Map` component for QML is the only way Qt provides for mapping, so you will have to use Qt Quick.

The `Map` component can be backed by various backend plugins. In fact, you need to specify which plugin you are using. `Map` has built-in support for the following plugins:

| **Provider** | **Key** | **Notes** | **Url** |
| Esri | esri | Subscription required | `www.esri.com` |
| HERE | here | Access token required | `developer.here.com/terms-and-conditions` |
| Mapbox | mapbox | Access token required | `www.mapbox.com/tos` |
| Mapbox GL | mapboxgl | Access token required | `www.mapbox.com/tos` |
| **Open Street Map** (**OSM**) | osm | Free access | `openstreetmap.org/` |

I will be using the OSM and HERE providers.

The HERE plugin requires an account at `developer.here.com`. It's easy to sign up and there is a free version. You need the app ID and app code to access their maps and API.

# Map

To start using the `Map` component, in your chosen `.qml` file, add both `QtLocation` and `QtPositioning` in the `import` lines:

```cpp
import QtLocation 5.12
import QtPositioning 5.12
```

The source code can be found on the Git repository under the `Chapter08-4` directory, in the `cp8` branch.

The `Map` component needs a `plugin` object, whose `name` property is one of the keys from the preceding table. You can set where the map is centered by setting the `center` property to a coordinate.

I am using the OSM backend and it is centered on the Gold Coast, Australia:

```cpp
    Map {
        anchors.fill: parent
        plugin: Plugin {
            name: "osm" 
        }
        center: QtPositioning.coordinate(-28.0, 153.4)
        zoomLevel: 10
    }
```

The Map centers on the coordinates we indicate with the `center` property, which is used to position the map to the user's current location.

We defined the Map's `plugin` property to be the `"osm"` plugin, which is an identifier for the Open Street Map plugin.

It is that easy to show a map. 

# MapCircle

You can highlight an area by placing a `MapCircle` in the `Map`. Again, centered on the Gold Coast.

A `MapCircle` has a `center` property that we can define by using a `latitude` and `longitude` location in a signed decimal value.

The `radius` property here is in the unit of meters according to the map. So in our example, the `MapCircle` will have a radius of 5,000 meters.

```cpp
        MapCircle {
             center {
                 latitude: -28.0
                 longitude: 153.4
             }
             radius: 5000.0
             border.color: 'red'
             border.width: 3
             opacity: 0.5
         }
```

Each map backend has its own parameters, which can be set using the `PluginParameter` component in the `Map` component. 

# PluginParameter

By default, the OSM backend downloads lower-resolution tiles. If you want high-resolution maps, you can specify the `'osm.mapping.highdpi_tiles'` parameter:

```cpp
PluginParameter {
    name: 'osm.mapping.highdpi_tiles'
    value: true
}
```

Each `PluginParameter` element holds just one `name`/`value` parameter pair. If you need to set several parameters, you will need a `PluginParameter` element for each:

```cpp
PluginParameter { name: "osm.useragent"; value: "Mobile and Embedded Development with Qt5"; }
```

Other `PluginParameters` you could consider are tokens and app IDs for various map providers, such as HERE maps.

Here's how our map looks, running on Android:

![](img/49538139-169e-48a8-8e2c-0309ad29cccf.png)

There are other Qt Quick items that we can use with addresses on the map. Let's look at routing.

# RouteModel

To show a route on a map, you will need to use `RouteModel`, which is a property of the `Map` item, `RouteQuery` to add waypoints, and a `MapItemView` to display it.

`RouteModel` needs a plugin, so we just reuse the plugin for the `Map` item. It also needs a `RouteQuery` for its `query` property:

```cpp
        RouteQuery {
            id: routeQuery
        }
        RouteModel {
            id: routeModel
            plugin : map.plugin
            query: routeQuery
        }
```

`MapItemView` is used to display model data on the map. It also needs a delegate of `MapRoute`. In our case, this is a line that describes the route:

```cpp
        MapItemView {
            id: mapView
            model: routeModel
            delegate: routeDelegate
        }
        Component {
            id: routeDelegate
            MapRoute {
                id: route
                route: routeData
                line.color: "#46a2da"
                line.width: 5
                smooth: true
                opacity: 0.8
            }
        }
```

Now what we need is a starting point, an ending point, and any point in between. In this example, I keep it simple and only specify start and end points. You can specify a GPS coordinate by using `QtPositioning.coordinate`, which takes a latitude and longitude value as arguments:

```cpp
​property variant startCoordinate: QtPositioning.coordinate(-28.0, 153.4)
property variant endCoordinate: QtPositioning.coordinate(-27.579744, 153.100175)
```

The start-point coordinate is some random area on the Gold Coast, Australia; the endpoint is where the last south-of-the-equator Trolltech office was. The `RouteQuery travelModes` property determines how the route is figured, whether traveling by car, foot, or public transport. It can be one of the following values:

*   `CarTravel` 
*   `PedestrianTravel`
*   `BicycleTravel`
*   `PublicTransit`
*   `TruckTravel`

The `RouteQuery` property, `routeOptimzations`, limits the query to the following different values:

*   `ShortestRoute`
*   `FastestRoute`
*   `MostEconomicRoute`
*   `MostScenicRoute`

In this example, I made `routeQuery` fire off in the `Component.onCompleted` signal. Usually, something like this would be triggered after the user has configured the query:

```cpp
Component.onCompleted: {
    routeQuery.clearWaypoints();
    routeQuery.addWaypoint(startCoordinate)
    routeQuery.addWaypoint(endCoordinate)
    routeQuery.travelModes = RouteQuery.CarTravel
    routeQuery.routeOptimizations = RouteQuery.FastestRoute
    routeModel.update();
 }
```

Here is how the route looks. This route indicated by the blue line starting in the big red circle:

![](img/189324f1-d058-4353-a244-4acaed6350f0.png)

You can add more `Waypoints` to establish different routes or get turn-by-turn directions by setting `routeModel` to a `ListView` or similar.

Not only can Qt Location show maps and routes, but there is also support for displaying places of interest, such as restaurants, gas stations, and national parks, in the `Places` API.

# Places of interest

At this point, I am going to switch to the HERE maps plugin. I tried to get the OpenStreetMaps places to work, but it could not find anything around.

In the next step of the construction of our map, we use `PlaceSearchModel` to search for places. As with the `RouteModel` before, `MapItemView` can display this model on our map.

Just like `RouteModel`, `PlaceSearchModel` needs some way of displaying the data; we could choose a `ListView`, which is useful for some purposes, but let's choose `MapItemView` for the visual effect.

We need to state which plugin we are using with `searchArea` and `searchTerm`:

```cpp
PlaceSearchModel {
    id: searchModel
    plugin: mapPlugin
    searchTerm: "coffee"
    searchArea: QtPositioning.circle(startCoordinate)
    Component.onCompleted: update()
}
```

Our `MapItemView` and `delegate` code look like this. The `searchView` delegate will show up as an icon with its title text, from the resulting place :

```cpp
MapItemView {
    id: searchView
    model: searchModel
    delegate: MapQuickItem {
        coordinate: place.location.coordinate
        anchorPoint.x: image.width * 0.5
        anchorPoint.y: image.height
        sourceItem: Column {
            Image { id: image; source: "map-pin.png" }
            Text { text: title; font.bold: true; color: "red"}
        }
    }
}
```

As you can see here, the place points are a bit difficult to read and are superimposed on top of the other ones that are around. This indicates that there are places too near each other for the zoom level and the map is having difficulties in placing the names. You can work around this issue by using a different zoom level or by using some collision detection and layout algorithms that I won't go into there.

![](img/90adbe26-04ad-42a1-a2a8-680274dde9d2.png)

The `map-pin.png` icon is from [https://feathericons.com/](https://feathericons.com/) and is released under the open source MIT license.

# Summary

In this chapter, we covered many aspects of mapping using Qt Location and Qt Positioning. We can get satellite information with `QGeoSatelliteInfo`, and locate the exact current position coordinates with `QGeoPositionInfo`. We learned how to use Qt Quick `Map` and different map providers to show the current location. We covered how to provide a route with `RouteModel`, search for places nearby using `PlaceSearchModel`, and show them using `MapItemView`.

In the next chapter, we will discuss audio and video with Qt Multimedia.